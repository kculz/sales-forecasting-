from flask import Flask, render_template, url_for, request, session,redirect
from flask_pymongo import PyMongo
import bcrypt

app = Flask(__name__)

app.config['MONGO_DBNAME']= 'sales-forecast'
app.config['MONGO_URI'] ='mongodb+srv://peony:peony@cluster0.bpitxm8.mongodb.net/sales-forecast'

mongo = PyMongo(app)
from main.models import User

@app.route('/user/signup', methods=['POST'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'username': request.form['username'], 'password': hashpass})
            return redirect(url_for('signup'))
    return 'User Already Exists'

@app.route('/')
def home():
    if 'username' in session:
        return session['username'] + ' Logged in'
    return render_template('dashboard.html')

@app.route('/login')
def login():
    users = mongo.db.users
    login_user = users.find_one({'username': 'admin'})
    if login_user:
        if bcrypt.hashpw(request.form['username'].encode('utf-8'), login_user['password'].encode('utf-8')) == login_user['password'].encode('utf-8'):
            session['username'] = request.form['username']
            return redirect(url_for('dashboard'))
    return 'Invalid Username || Password'

if __name__ == '__main__':
    app.run(debug=True)