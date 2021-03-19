class SQL:
    
    def __init__(self, host,user,pw,db,table):
        self.host = host
        self.user = user
        self.pw = pw
        self.db = db
        self.table = table

    def connect(self):
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
            #data = cursor.execute("SELECT * FROM aug_train")
            query = "SELECT * FROM " + self.table
            df = pd.read_sql(query,con=conn)
            return df
        except Error as e:
            print(e)
    
    #def save_data(self):
        # Read data into DB

        
    def query_data(self):
        #Query data from db
        self.connect.execute("SELECT * FROM "+ self.table)


c = SQL("localhost","root",123456,"hr_analytics","aug_train")
c= c.connect()
print(c)
