from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a46506038145' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key= True)
    username = db.Column(db.String(20), nullable= False)
    password = db.Column(db.String(80), nullable= False)

with app.app_context():
    db.create_all()
    
@app.route('/')
def home():
    return "hello from flask app"

if __name__ == '__main__':
    app.run(debug=True)