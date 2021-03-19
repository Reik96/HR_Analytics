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
        #Insert data to database
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
            query = "INSERT INTO" + self.table +"(looks_for_job) VALUES (%s)"
            cursor.executemany(query,predictions)
            conn.commit()
            msg = print(cursor.rowcount, "was inserted.")
            return msg
        except Error as e:
            print(e)

#c = SQL("localhost","root",123456,"hr_analytics","aug_train")
#c= c.query_data()
#print(c)

