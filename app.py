from flask import Flask, render_template, url_for, request, redirect,session
import sqlite3
from forecast import monthly_sales, predict_data_frame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
import io
import seaborn as sns

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        connection = sqlite3.connect('sales.db')
        cursor = connection.cursor()

        username = request.form['username']
        password = request.form['password']
        query = "SELECT username, password FROM users WHERE username= '"+username+"' AND password='"+password+"' "
        cursor.execute(query)
        result = cursor.fetchall()
        if len(result) == 0:
            return "Invalid username || password"
        else:
            return render_template('dashboard.html')
    return render_template('index.html')


@app.route('/forecast')
def forecast():
    fig = plt.figure(figsize=(15,5))
    #Actual sale
    plt.plot(monthly_sales['date'], monthly_sales['sales'])
    #Predicted sales
    plt.plot(predict_data_frame['date'], predict_data_frame['Linear Prediction'])
    plt.title("sales forecast using Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(['Actual Sales', 'Predicted Sales'])
    canvas = FigureCanvas(fig) 
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url =base64.b64encode(img.getbuffer()).decode('utf-8')
    # print(type(plot_url))
    return render_template('forecast.html', plot_url=plot_url)

@app.route('/upload')
def upload():
    print("hello")
    return render_template('upload.html')

@app.route('/createUser', methods=['GET','POST'])
def createUser():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        connection = sqlite3.connect('sales.db')
        cursor = connection.cursor()
        query = "INSERT INTO users (username, password) VALUES('"+ username +"', '"+ password +"')"

        if cursor.execute(query):
            msg = "User added!"
        else:
            connection.rollback()
            msg = "Insertion error!"

        connection.close()
        return render_template('users.html', msg=msg)

@app.route('/user')
def user():

    connection = sqlite3.connect('sales.db')
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    query = "SELECT * FROM users"
    cursor.execute(query)
    users = cursor.fetchall()
    
    return render_template('users.html', users=users)



if __name__ == '__main__':
    app.run(debug=True)