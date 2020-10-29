from flask_wtf import Form
from wtforms import RadioField, SubmitField, SelectField, TextAreaField

from wtforms import validators, ValidationError
from wtforms.validators import DataRequired, Optional

#create conditional statements
def only_if(form, field):
	if (form.kidney.data!='invalid'):
		raise ValidationError('You can answer this if the organ was marked as invalid')
def condition(form, field):
    if form.relevant.data!='Y':
        raise ValidationError('You can only answer this question if the previous answer was yes!')

#form to be filled to label the data
class GoalsForm(Form):
	#set variables in the Form
	kidney= RadioField('Is this review POSITIVE', choices=[('1','positive'),('0','negative'),('invalid','invalid report')], validators=[DataRequired()])
	description=TextAreaField('If this text is not relevant. Describe why:', validators=[Optional(), only_if])
	initial_time=TextAreaField('Initial Time')
	send=SubmitField(label='Send')

#form for login page
class login(Form):
    names=RadioField('names', choices=[('User','Uset')],
                     validators=[DataRequired()])
    submit=SubmitField(label='Submit')
#form for pages between reports
class Time_counter(Form):
    submit=SubmitField(label='Start')
    
	
	