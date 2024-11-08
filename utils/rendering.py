import pandas as pd
import numpy as np
from gym_trading_env.renderer import Renderer
from flask import Flask, render_template, jsonify, make_response
import os
from pyecharts.globals import CurrentConfig
from pyecharts import options as opts
from pyecharts.charts import Bar

from gym_trading_env.utils.charts import charts
from pathlib import Path 
import glob
from cachetools import TTLCache
import sqlite3 as db


class LiveRenderer(Renderer):
    def __init__(self, render_logs_dir,quote_asset="USD",base_asset="ETH"):
        super().__init__(render_logs_dir)
        self.app = Flask(__name__, static_folder="./templates/")
        
        self.add_metric(f'Quote Balance:',lambda df : f"{(df['portfolio_distribution_fiat'].values[-1]):0.2f} {quote_asset} ")
        self.add_metric(f'Asset Balance:',lambda df : f"{(df['portfolio_distribution_asset'].values[-1]):0.2f} {base_asset}")
        self.add_metric(f'Total Valuation:',lambda df : f"{(df['portfolio_valuation'].values[-1]):0.2f} {quote_asset}")
        
    def connect_to_db(self):
        conn = db.connect(self.render_logs_dir)
        return conn
    

    def get_data(self,conn,name=None):
        
        #####################################################################
        #start of part that I need to refresh
        data=pd.read_sql(f'select * from {name}',conn)
        if 'date' not in list(data.columns):
            data['date']=data['ds'].copy()
        return data

    
    def run(self,**kwargs):
        
        @self.app.route("/")
        def index():
            conn=self.connect_to_db()
            cursor=conn.cursor()
            sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
            render_names =cursor.execute(sql_query)
            
            render_names=[t[0] for t in cursor.fetchall()]
            render_names=[name for name in render_names if 'trade' in name]
            
            return render_template('test_net.html', render_names = render_names)

        @self.app.route("/update_data/<name>")
        def update(name = None):
            conn=self.connect_to_db()
            self.df=self.get_data(name=name,conn=conn)
            self.df['date']=self.df['date'].apply(pd.to_datetime)
            self.df=self.df.set_index('date')
            chart = charts(self.df, self.lines)
            return chart.dump_options_with_quotes()

        @self.app.route("/metrics")
        def get_metrics():
            self.compute_metrics(self.df)
            return jsonify([{'name':metric['name'], 'value':metric['value']} for metric in self.metrics])

        self.app.run(**kwargs)

    