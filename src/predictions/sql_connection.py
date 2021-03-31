class SQL:
    """SQL Class that connects to desired database to query and insert data.
        Input arguments are host, user, password and database."""
    def __init__(self, host,user,pw,db,table):
        self.host = host
        self.user = user
        self.pw = pw
        self.db = db
        self.table = table

    def query_data(self,latest = False):
        # connect with the desired DB
        import mysql.connector
        from mysql.connector import MySQLConnection,Error
        import pandas as pd
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=str(self.pw),
                db = self.db,
            )
            cursor = conn.cursor(buffered=True)
            query = "SELECT * FROM " + self.table 
           
            if latest == True:
                query= " SELECT * FROM hr_analytics.aug_train WHERE  enrollee_id NOT IN (SELECT enrollee_id FROM hr_analytics.predictions)"
           
            df = pd.read_sql(query,con=conn)
            conn.close()
            cursor.close()
            return df
        except Error as e:
            print(e)
  
    def insert_data(self,predictions):
        from mysql.connector import MySQLConnection,Error
        import pandas as pd
        import numpy as np
        import pymysql
        from sqlalchemy import create_engine
        
        try:
            
            engine = create_engine("mysql+pymysql://"+ self.user + ":" + str(self.pw) + "@" + self.host + "/" + self.db )
            predictions.to_sql(con=engine, name='predictions', if_exists='append',index =True)
            return print("Data stored in SQL")
        except Error as e:
            print(e)

if __name__ == "__main__":
   SQL()