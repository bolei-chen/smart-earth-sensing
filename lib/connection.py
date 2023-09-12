import mysql.connector

class SQL_Connection:

    def __init__(self, __host, __user, __password, __database):
        self.db = mysql.connector.connect(
            host=__host,
            user=__user,
            password=__password,
            database=__database
        )
        self.cursor = self.db.cursor() 

    def execute(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def store_file(self, info):
        try:
            query = "INSERT INTO behavior (status, create_time) VALUES (%s, %s)"
            self.cursor.execute(query, info)
            self.db.commit()
            return True
        except Exception as e:
            print(f"error: {str{e}}") 
            return False
    def close(self):
        self.cursor.close()
        self.db.close() 
             