import pandas as pd
#db=pd.read_csv('db.csv')
import numpy as np
import random
import math
import pickle
import ast
import re
def length_db():
	db=pd.read_csv('database.csv', encoding="latin-1")
	return len(db)

def data_to_dict(num):
	le=pd.read_csv('model/kidney/auc.csv')
	db=pd.read_csv('database.csv', encoding="latin-1")
	#db=data.sample(1)
	return {'Report': db.get('Report#').iloc[num], 'note_text':db.get('review').iloc[num],'labeled_reports':len(le)}
	
	
def exist_label(num):
	print ('row '+str(num))
	db=pd.read_csv('database.csv', encoding="latin-1")
	try:
		return (not math.isnan(db.get('label').iloc[num]))
	except TypeError:
		return True

def get_label(num):
	db=pd.read_csv('database.csv', encoding="latin-1")
	try:
		return { 'Kidneys': int(db.get('label').iloc[num])}
	except ValueError:
		return { 'Kidneys': ast.literal_eval(db.get('label').iloc[num])}

def add_output(form):
	return {'Kidneys':form.kidney.data,'note_clinician':form.description.data, 'initial_time':form.initial_time.data}

def reset(form):
	form.kidney.data=None
	form.description.data=None
 
 
def change_database_invalid(num, row):
	db=pd.read_csv('database.csv', encoding="latin-1")
	db.loc[num, 'label']=row['Kidneys']
	try:
		db.loc[num, 'note_clinician']=row['note_clinician']
	except:
		db.loc[num, 'note_clinician']='No notes from clinicians'
	db.loc[num, 'valid']=False
	db.loc[num, 'group']='active'
	db.loc[num,'determinants']=float('inf')
	db.to_csv('database.csv', index=False, encoding="latin-1")

def change_database_existing(num, row):
	db=pd.read_csv('database.csv', encoding="latin-1")
	db.loc[num, 'group']='active'
	ent=row['determinants']
	if type(ent)!=type(None):
		db.loc[db['valid'],'determinants']=ent
	db.loc[num, 'valid']=False
	db.loc[num, 'determinants']=float('inf')
	db.to_csv('database.csv', index=False)

def get_report():
	db=pd.read_csv('database.csv', encoding="latin-1")
	k=random.choice(list(db.loc[db['valid'],:].index))
	return k
	
	
def get_tokens(num):
	db=pd.read_csv('database.csv', encoding="latin-1")
	token=db.get('tokens_num').iloc[num]
	return token
def get_inputs(dict):
	return [int(dict['Kidneys'])]
	
def add_report_train(num, dict):
	token=ast.literal_eval(get_tokens(num))
	labels=get_inputs(dict)[0]
	file=pickle.load(open('model/kidney/train.p', 'rb'))
	file[0].append(token)
	file[2]=np.append(file[2], np.array([labels]), axis=0)
	file[-1]=max(file[-1], len(token))
	pickle.dump(file,open('model/kidney/train.p', 'wb'))
	
def get_report_entropy():
	db=pd.read_csv('database.csv')
	e=db['entropy'].values
	selection = (np.argsort(e)[::-1])[0]
	return selection
def get_report_determinant():
	db=pd.read_csv('database.csv', encoding="latin-1")
	e=db['determinants'].values
	selection = (np.argsort(e))[0]
	return selection

def extract_findings(note):
	c=re.compile(r'(?i)findings:')
	d=re.compile(r'(?i)impression:')
	start=0
	end_=-1
	if bool(c.search(note)):
		start=re.search(c, note).start()
	return note[start:]