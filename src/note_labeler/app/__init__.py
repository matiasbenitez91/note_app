#from config import Config
#app.config.from_object(Config)
#app.debug=True



from flask import Flask
app=Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'
app.config['DEBUG'] = False

from note_labeler.app import routes
