from flask import Flask, render_template

app = Flask(__name__)

from main.models import User

@app.route('/user/signup', methods=['POST'])
def signup():
    return User().signup() 

@app.route('/')
def home():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)