class SQL:
    
    def __init__(self, host,user,pw,db,table):
        self.host = host
        self.user = user
        self.pw = pw
        self.db = db
        self.table = table

    def query_data(self):
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
            df = pd.read_sql(query,con=conn)
            conn.close()
            cursor.close()
            return df
        except Error as e:
            print(e)
  
    def insert_data(self,predictions):

        self.predictions = predictions.astype(int)
        #self.predictions = predictions.tolist()
        #Insert data to database
        import mysql.connector
        from mysql.connector import MySQLConnection,Error
        import pandas as pd
        import numpy as np

        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=str(self.pw),
                db = self.db,
            )
            cursor = conn.cursor()
            query = "INSERT INTO " + str(self.table) +"(looking_for_job) VALUES (%s)"
            #pred = self.predictions.tolist()
            pred = ("1","0")

            cursor.executemany(query,pred)
           # prediction_value= cursor.fetchall()
            conn.commit()
            
            msg = print(cursor.rowcount, "was inserted.")
            cursor.close()
            #return print(prediction_value)
            return print(self.predictions)
        except Error as e:
            print(e)

#from notebook.model_evaluation import y_pred
#c = SQL("localhost","root",123456,"hr_analytics","aug_train")
#c= c.query_data()
#c = SQL("localhost","root",123456,"hr_analytics","aug_test")
#c= c.insert_data(y_pred)
#print(c)

if __name__ == "__main__":
   SQL()