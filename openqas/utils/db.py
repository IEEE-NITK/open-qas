import sqlite3

DEFAULT = "wiki.db"

class WikiDB:
    def __init__(self, path=DEFAULT):
        self.connection = sqlite3.connect(path)
        self.path = path
    
    def get_all_doc_ids(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def close(self):
        self.connection.close()
    
    def get_doc_text(self, id):
        cursor = self.connection.cursor()
        id = (id, )
        cursor.execute("SELECT text FROM documents WHERE id = ?", id)
        result = cursor.fetchone()
        cursor.close()
        return result[0]

    def get_doc_title(self, id):
        cursor = self.connection.cursor()
        id = (id, )
        cursor.execute("SELECT title FROM documents WHERE id = ?", id)
        result = cursor.fetchone()
        cursor.close()
        return result[0]

    def get_all_doc_texts(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents")
        result = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return result
