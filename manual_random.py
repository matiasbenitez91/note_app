#from app import app
#from flask import render_template, request, flash, redirect, url_for
from app.fun import *
from app.forms import GoalsForm, login
from model.get_auc import *

def index():	
	import pandas as pd
	from time import gmtime, strftime
	#form=GoalsForm(request.form)
	user_output="data/output.csv"
	#user_count="data/"+str(user)+".txt"
	"""with open(user_count, 'r') as f:
		read=f.readlines()
		num=int(read[0])
		end=int(read[1])
	"""
	#num=get_report_entropy()
	try:
		num=get_report_determinant()
		
	except:
		print ('error entropy')
		return False
	note=data_to_dict(num)
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
		out_auc=out_auc.append({'Report#':row['Report#'], 'AUC_test':auc_, 'AUC_val':max_auc, 'time':strftime("%Y-%m-%d %H:%M:%S", gmtime())}, ignore_index=True)
		out_auc.to_csv('model/kidney/auc.csv', index=False)
		df=pd.DataFrame({'AUC_test': [auc_],'AUC_val':[max_auc]})
		df.to_csv('model/kidney/max_auc.csv', index=False)#form=GoalsForm(formdata=None)
		change_database_existing(num, row)
		return True
	else:
		return False
		
		
boolean=True
while boolean:
	boolean=index()
