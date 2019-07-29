import pandas as pd
from time import gmtime, strftime
from app import app
from flask import render_template, request, flash, redirect, url_for
from app.fun import *
from app.forms import GoalsForm, login, Time_counter
from model.get_auc import *


@app.route('/ajsjalaksneeeoeoa284744635421352a41fsoijf;aljxk;ioefm', methods=['GET', 'POST'])
def index():
	initial_time=request.args.get('initial_time')
	form=GoalsForm(request.form)
	if request.method == 'GET':
		form.initial_time.data = initial_time
	user_output="data/output.csv"
	try:
		num=get_report_determinant()
	except:
		return redirect(url_for('end'))
	note=data_to_dict(num)
	note['note_text']=extract_findings(note['note_text'])
	if exist_label(num):
		row=get_label(num)
		row['Report#']=note['Report']
		print ('NOTE EXISTING	'+str(note['Report']))
		row['time']=strftime("%Y-%m-%d %H:%M:%S", gmtime())
		output=pd.read_csv(user_output)
		output=output.append(row, ignore_index=True)
		output.to_csv(user_output, index=False)

		add_report_train(num, row)
		auc_, max_auc, determinants=main()
		row['determinants']=determinants
		out_auc=pd.read_csv('model/kidney/auc.csv')
		out_auc=out_auc.append({'Report#':row['Report#'], 'AUC_test':auc_, 'AUC_val':max_auc, 'final_time':strftime("%Y-%m-%d %H:%M:%S", gmtime())}, ignore_index=True)
		out_auc.to_csv('model/kidney/auc.csv', index=False)
		df=pd.DataFrame({'AUC_test': [auc_],'AUC_val':[max_auc]})
		df.to_csv('model/kidney/max_auc.csv', index=False)
		change_database_existing(num, row)
		return redirect(url_for('template_time'))
	else:
	#x=note['data']
	##new code here
		if form.validate_on_submit():
			initial_time=form.initial_time.data
			final_time=strftime("%Y-%m-%d %H:%M:%S", gmtime())
			row=add_output(form)
			print("initial time", initial_time)
			print("final time", final_time)
			#print(request.args)
			if row['Kidneys']!='invalid':
				row['Report#']=note['Report']
				row['time']=strftime("%Y-%m-%d %H:%M:%S", gmtime())
				output=pd.read_csv(user_output)
				output=output.append(row, ignore_index=True)
				output.to_csv(user_output, index=False)
				add_report_train(num, row)
				auc_, max_auc, determinants=main()
				row['determinants']=determinants
				out_auc=pd.read_csv('model/kidney/auc.csv')
				out_auc=out_auc.append({'Report#':row['Report#'], 'AUC_test':auc_, 'AUC_val':max_auc, 'initial_time':initial_time, "final_time": final_time}, ignore_index=True)
				out_auc.to_csv('model/kidney/auc.csv', index=False)
				df=pd.DataFrame({'AUC_test': [auc_],'AUC_val':[max_auc]})
				df.to_csv('model/kidney/max_auc.csv', index=False)#form=GoalsForm(formdata=None)
				change_database_existing(num, row)
			else:
				row['Report#']=note['Report']
				row['time']=strftime("%Y-%m-%d %H:%M:%S", gmtime())
				output=pd.read_csv(user_output)
				output=output.append(row, ignore_index=True)
				output.to_csv(user_output, index=False)
				change_database_invalid(num, row)

			return redirect(url_for('template_time'))
	reset(form)
	return render_template('index.html', note=note, form = form)

@app.route('/alksjdnald')
def end():
	return render_template('end.html')
	
@app.route('/time;jkfna;sknlfas;kfna;skjfna;iowhre198p4yt4;', methods=['GET','POST'])
def template_time():
	form=Time_counter(request.form)
	if form.validate_on_submit():
		time=strftime("%Y-%m-%d %H:%M:%S", gmtime())
		print (request.args)
		return redirect(url_for('index', initial_time=time))
	return render_template('time.html', form=form)

@app.route('/', methods=['GET', 'POST'])
def login():
	error = None
	if request.method == 'POST':
		if request.form['username'] != 'user' or request.form['password'] != '123456':
			error = 'Invalid Credentials. Please try again.'
		else:
			return redirect(url_for('template_time'))
	return render_template('login.html', error=error)
