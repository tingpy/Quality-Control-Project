# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:34:36 2021

@author: acer
"""

from itertools import compress
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import pearsonr
from pystan import StanModel
#import pystan

import os 
from os import listdir
from os.path import isfile, join
import datetime
import json

def isset(var):
    try:
        a = type(eval(var))
    except:
        return False
    return a

def import_data(data_location):
    data_location = 'C:\\Users\\acer\\Desktop\\Chimei\\QC data'
    data_location = data_location + '\\raw_data'
    os.chdir(data_location)
    
    # for spec data
    drop_col_name = ['MG 2', 'MG 1', '參考編號']
    standard_col_name = ['material', 'LOTNO', 'level', 'spec name', 'spec value',
                          'factory', 'line', 'date', 'spec limit']
    
    spec_file = [f for f in listdir('.\\spec') if isfile('.\\spec\\' + f)]
    spec_inform = []
    for file in spec_file:
        spec_tmp = pd.read_csv(data_location + '\\spec\\' + file,
                                encoding = 'utf-8')
        for drop_column in drop_col_name:
            try:
                spec_tmp = spec_tmp.drop([drop_column], axis=1)
            except:
                print(drop_column + '''isn't removed from ''' + file)
        spec_tmp.columns = standard_col_name
        spec_inform.append(spec_tmp)
    
    spec_inform = pd.concat(spec_inform)
    
    ## import deal data
    # 'deal inform' is the dataframe that stores all deal data
    drop_col_name = ['DN', '交貨', '項目', '貨櫃號碼', 'STATUS', 'ITYPE', 'WERKS', '重量']
    standard_col_name = ['level', 'shipment NO', 'material', 'LOTNO', 'buyer',
                         'receiver', 'date']
    
    deal_file = [f for f in listdir('.\\deal') if isfile('.\\deal\\' + f)]
    deal_inform = []
    for file in deal_file:
        deal_tmp = pd.read_csv(data_location + '\\deal\\' + file,
                               encoding = 'utf-8')
        for drop_column in drop_col_name:
            try:
                deal_tmp = deal_tmp.drop([drop_column], axis=1)
            except:
                print(drop_column + '''isn't removed from ''' + file)
        deal_tmp.columns = standard_col_name
        deal_inform.append(deal_tmp)
     
    deal_inform = pd.concat(deal_inform)

    
    ## variable 'date' from string to time
    deal_inform['date'] = pd.to_datetime(deal_inform['date'])
    spec_inform['date'] = pd.to_datetime(spec_inform['date'])
    
    ## substring the material context to 7 characters
    material_brief = deal_inform['material'].array
    for i in range(0, len(deal_inform)):
        # in case that there will be blanks in front of the characters
        try:
            for j in range(0,100):
                if material_brief[i][j] != ' ':
                    break
        except: # skip NA
            continue
        
        try:
            material_brief[i] = material_brief[i][j:(j+7)]
        except:
            material_brief[i] = material_brief[i][j:]
    deal_inform['material'] = material_brief
    
    material_brief = spec_inform['material'].array
    for i in range(0, len(spec_inform)):
        try:
            for j in range(0,100):
                if material_brief[i][j] != ' ':
                    break
        except: # skip NA
            continue
        
        try:
            material_brief[i] = material_brief[i][j:(j+7)]
        except:
            material_brief[i] = material_brief[i][j:]
    spec_inform['material'] = material_brief
    
    # information of agent(代理商) and customer name
    agent_file = [f for f in listdir('.\\agent') if isfile('.\\agent\\' + f)]
    agent_df = []
    for file in agent_file:
        agent_tmp = pd.read_csv(data_location + '\\agent\\' + file,
                                encoding = 'utf-8')
        agent_df.append(agent_tmp)
    if(len(agent_df) == 1):
        agent_df = agent_df[0]
    else:
        agent_df = pd.concat(agent_df, axis=1)
        
    customer_file = [f for f in listdir('.\\customer') if isfile('.\\customer\\' + f)]
    customer_df = []
    for file in customer_file:
        customer_tmp = pd.read_csv(data_location + '\\customer\\' + file,
                                   encoding = 'utf-8')
        customer_df.append(customer_tmp)
    if(len(customer_df) == 1):
        customer_df = customer_df[0]
    else:
        customer_df = pd.concat(customer_df, axis=1)
    
    # extract the useful information at the dataframe
    string_bool = []
    for i in agent_df['買方客戶']:
        if '合計' not in i:
            string_bool.append(True)
        else:
            string_bool.append(False)
    agent_df = agent_df.iloc[string_bool, :]
    
    return spec_inform, deal_inform, agent_df, customer_df


def name_new_csv(time_stamp):
    year = time_stamp.year
    month = time_stamp.month
    date = time_stamp.day
    
    return str(year) + '_' + str(month) + '_' + str(date)

# def file_cnt_detect(name):
#     files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#     subs = '_stan_result'
#     subs2 = '.json'
#     files = [i for i in files if subs in i]
#     model_num = [int(i.split(subs)[1].split(subs2)[0]) for i in files]
    
#     if model_num == []:
#         count = 0
#     else:
#         count = max(model_num)
        
#     count = count + 1


def LOT_based_df(spec_interest, select_spec, data_type, return_delete_inform):
    group_obj = spec_interest.groupby('spec name') 
    store_series = []
    store_duplicate_LOTNO = []
    df_date = pd.DataFrame(data = None, columns = ['date'])
    for k, v in group_obj:
        print(k)
        temp_series = pd.Series(v['spec value'].tolist(), name=k)
        temp_series.index = v['LOTNO']
        temp_series = temp_series.dropna()
        
        # remove the duplicated LOTNO
        temp_dup = list(temp_series.loc[temp_series.index.duplicated(),].index)
        temp_series = temp_series.drop(temp_dup)
        for j in temp_dup:
            if j not in store_duplicate_LOTNO:
                store_duplicate_LOTNO.append(j)
                
        store_series.append(temp_series)
        
        temp_date_df = pd.DataFrame({'date': v['date'].tolist()}, 
                                    index = v['LOTNO'])
        temp_date_df = temp_date_df[~temp_date_df.index.duplicated(keep='first')]
        df_date = df_date.combine_first(temp_date_df)
        

    df = pd.concat(store_series, axis = 1)
    
    df = df.sort_index(ascending=True)
    df_date = df_date.sort_index(ascending=True)
    lotno_inform = pd.concat([df_date, df], axis=1)
    lotno_inform.insert(loc=0, column='LOTNO', value=lotno_inform.index)
    lotno_inform.index = [i for i in range(0, lotno_inform.shape[0])]

    if isinstance(select_spec, list):
        reserve_col_name = ['LOTNO', 'date'] + select_spec
    else:
        reserve_col_name = ['LOTNO', 'date'] + [select_spec]

    drop_col = []
    for i in lotno_inform.columns:
        if i not in reserve_col_name:
            drop_col.append(i)
    lotno_inform = lotno_inform.drop(drop_col, axis=1)
    
    if return_delete_inform is True:
        return lotno_inform, store_duplicate_LOTNO
    
    lotno_inform = lotno_inform.dropna(axis = 0)
    if data_type == 'Standardized':
        for i in range(2, lotno_inform.shape[1]):
            lotno_inform.iloc[:, i] = preprocessing.scale(
                lotno_inform.iloc[:, i].values)
            
    return lotno_inform
    


def spec_to_dic(spec_interest, select_spec, data_type):
    store_dic = {}
    col_loc = spec_interest.columns == 'spec value'
        
    for i in select_spec:
        df_bool = spec_interest['spec name'] == i
        spec_df = spec_interest.loc[df_bool, col_loc]
        spec_df = spec_df.dropna()

        if data_type == 'Standardized':
            store_dic[i] = preprocessing.scale(spec_df.T.values[0])
        else:
            store_dic[i] = spec_df.T.values[0]
            
    return store_dic

def time_shift(spec_interest, select_spec, data_type):
    store_dic = {}
    col_loc = spec_interest.columns == 'spec value'
    df = spec_interest[spec_interest['spec name'] == select_spec[0]]
    spec_limit = set(df['spec limit'])
  
    for i in spec_limit:
        df_bool = df['spec limit'] == i
        spec_df = df.loc[df_bool,:]
        max_date = spec_df['date'].max()
        min_date = spec_df['date'].min()
        max_date = str(max_date.year) + '/' + str(max_date.month)
        min_date = str(min_date.year) + '/' + str(min_date.month)
        spec_df = spec_df.loc[:, col_loc]
        spec_df = spec_df.dropna()
        
        
        dic_name = i + ' (' + min_date + ' - ' + max_date + ')'
        if data_type == 'Standardized':
            store_dic[dic_name] = preprocessing.scale(spec_df.T.values[0])
        else:
            store_dic[dic_name] = spec_df.T.values[0]
            
    return store_dic

def corr_var(focus_spec_name, lot_based_df):
    corr_df = pd.DataFrame(data = None, 
                           columns = lot_based_df.columns)
    
    for i in range(0, len(focus_spec_name)):
        corr_vector = []
        focus_bool = lot_based_df.columns == focus_spec_name[i]
        focus_spec_value = list(lot_based_df.loc[:, focus_bool].T.iloc[0,:])
        focus_na_bool = np.isnan(focus_spec_value)
        
        for j in lot_based_df.columns:
            if j not in focus_spec_name:
                other_bool = lot_based_df.columns == j
                other_spec_value = list(lot_based_df.loc[:, other_bool].T.iloc[0,:])
                other_na_bool = np.isnan(other_spec_value)
                
                corr_value = pearsonr(np.array(focus_spec_value)[~np.logical_or(focus_na_bool,other_na_bool)],
                                      np.array(other_spec_value)[~np.logical_or(focus_na_bool,other_na_bool)])
                if np.isnan(corr_value[0]):
                    corr_vector.append(0)
                else:
                    corr_vector.append(abs(corr_value[0]))
            else:
                corr_vector.append(0)
        corr_df.loc[i] = corr_vector
    
    corr_df.index = focus_spec_name
    
    return corr_df


def recommend_var(cor_df, max_var_cnt):
    input_var_df = {}
    var_store = []
    
    for i in cor_df.columns.values:
        temp_col = list(cor_df.sort_values(i, ascending=False).index)
        if len(var_store) == 0:
            var_store = var_store + temp_col[0:max_var_cnt]
            input_var_df[i] = temp_col[0:max_var_cnt]
        else:
            cnt = 0
            more_store = []
            for j in range(0, len(cor_df.index)):
                if (temp_col[j] not in var_store) & (temp_col[j] != i):
                    var_store.append(temp_col[j])
                    more_store.append(temp_col[j])
                    cnt = cnt + 1
                if cnt == max_var_cnt:
                    break
            input_var_df[i] = more_store

    return input_var_df

def update_recommend_var(cor_df, new_dic, old_dic, max_var_cnt): 
    cor_df.index = cor_df['Unnamed: 0']
    cor_df = cor_df.drop('Unnamed: 0', axis=1)
    
    focus_var = cor_df.columns.values
    old_removed_var = {}
    for i in range(len(focus_var)):
        try:
            old_removed_var[focus_var[i]] = new_dic[i].split(', ')
        except:
            old_removed_var[focus_var[i]] = []
    
    var_store = []
    for i in focus_var:
        index_name = list(cor_df.sort_values(i, ascending=False).index)
        last_element_index = index_name.index(old_dic[i][len(old_dic[i])-1]) + 1
        var_store = var_store + index_name[0:last_element_index]
    var_store = var_store + list(cor_df.columns)
            
    new_input_var = {}
    for (k,i) in enumerate(focus_var):
        if new_dic[k] == '':
            new_input_var[i] = []
        else:
            new_input_var[i] = new_dic[k].split(', ')
        
        append_cnt = 0
        gap_cnt = max_var_cnt - len(new_input_var[i])
        if gap_cnt == 0:
            next
        else:
            temp_index = list(cor_df.sort_values(i, ascending=False).index)
            old_var = old_dic[i][len(old_dic[i])-1]
            last_element_index = temp_index.index(old_var)
            
            for j in range(last_element_index+1, len(temp_index)):
                if gap_cnt == append_cnt:
                    break
                else:
                    if temp_index[j] not in var_store:
                        new_input_var[i].append(temp_index[j])
                        var_store.append(temp_index[j])
                        append_cnt = append_cnt + 1
    
    return new_input_var
        

def spec_deal_concat(material, spec, deal, dic, type):
    spec_df = spec[spec['material'] == material]
    deal_df = deal[deal['material'] == material]
    
    reserve_spec = []
    for i in list(dic.keys()):
        reserve_spec.append(i)
        if len(dic[i]) > 0:
            reserve_spec = reserve_spec + dic[i]
    
    unique_lotno = spec_df['LOTNO'].unique()
    reserve_deal_bool = [i in unique_lotno for i in deal_df['LOTNO'].values]
    deal_df = deal_df.loc[reserve_deal_bool, :]
    
    spec_df = LOT_based_df(spec_df, reserve_spec, 'Original', False)
    spec_df = spec_df.dropna(axis = 0)
    
    mean_vec = []
    sd_vec = []
    for i,j in enumerate(reserve_spec):
        mean_vec.append(np.mean(spec_df[j].values))
        sd_vec.append(np.std(spec_df[j].values))
        if type == 'Standardized':
            spec_df[j] = (spec_df[j].values - mean_vec[i]) / sd_vec[i]
    mean_var_dic = {'spec': reserve_spec, 'mean': mean_vec, 'sd': sd_vec}
        
    
    col_name = ['buyer', 'LOTNO', 'date', 'level']
    df_model = pd.DataFrame(data = None,
                            columns = col_name + reserve_spec)
    
    col_loc = []
    for i in ['buyer', 'LOTNO', 'date', 'level']: 
        col_loc.append(deal_df.columns.get_loc(i))
    row_cnt = 0
    for i in spec_df['LOTNO'].values:
        deal_lotno_index = np.where(deal_df['LOTNO'] == i)[0]
        try:
            buyers_df = deal_df.iloc[deal_lotno_index, col_loc]
        except:
            continue
        customer_cnt = len(buyers_df)    
        
        for j in reserve_spec:
            spec_col_loc = spec_df.columns.get_loc(j)
            spec_value = [spec_df.iat[row_cnt, spec_col_loc]] * customer_cnt
            buyers_df[j] = spec_value
        df_model = pd.concat([df_model, buyers_df])
        row_cnt = row_cnt + 1
        
    return df_model, mean_var_dic


    

quality_control = '''
data{
  int Nt;
  int Ns;
  matrix[Nt,Ns] spec;
}
parameters{
  vector[Ns] A_mu;
  vector[Ns] mu;
  vector<lower=0>[Ns] sigma;
  matrix[Ns,Ns-1] A;
}
model{
  
  for(i in 1:Ns){
    A[i,] ~ normal(0,1);
  }

  for(i in 1:Ns){
    A_mu[i] ~ normal(0,1);
  }
  
  for(i in 1:Ns){
    sigma[i] ~ lognormal(0,1);
  }
  
  for(i in 1:Nt){
    mu[1] ~ normal(A_mu[1] + dot_product(A[1,], spec[i, 2:Ns]), 5);
    spec[i, 1] ~ normal(mu[1], sigma[1]);
    for(j in 2:(Ns-1)){
      mu[j] ~ normal(A_mu[j] + dot_product(A[j, 1:(j-1)], spec[i, 1:(j-1)]) + 
                       dot_product(A[j, j:(Ns-1)], spec[i, (j+1):Ns]), 5);
      spec[i, j] ~ normal(mu[j], sigma[j]);
    }
    mu[Ns] ~ normal(A_mu[Ns] + dot_product(A[Ns,], spec[i,1:(Ns-1)]), 5);
    spec[i, Ns] ~ normal(mu[Ns], sigma[Ns]);
  }
  
}
'''

def run_stan(concat_df, dic,
              quality_control = quality_control):
    iter_cnt = 10000
    output_cnt = 2500
    
    reserve_spec = []
    for i in list(dic.keys()):
        reserve_spec.append(i)
        if len(dic[i]) > 0:
            reserve_spec = reserve_spec + dic[i]
    
    buyer_name = []
    for i in concat_df['buyer'].values:
        buyer_name.append(str(i))
    concat_df['buyer'] = buyer_name
    
    # posterior = pystan.StanModel(model_code=quality_control)
    posterior = StanModel(model_code=quality_control)
    
    unique_buyer = concat_df['buyer'].unique()
    customer_para = {}
    purchase_cnt = []
    customer_name_list = []
##################################################################
    # cnt1 = 0
##################################################################
    global ccc
    ccc = []
    for i in unique_buyer:
        if 'TW' in i:
            continue
        temp_df = concat_df[concat_df['buyer'] == i]
        
        # print(len(temp_df))
        # print(i)
        purchase_cnt.append(len(temp_df))
        customer_name_list.append(i)
        
        stan_data = {"Nt": len(temp_df),
                      "Ns": len(reserve_spec),
                      "spec": temp_df[reserve_spec]}
        try:
            fit = posterior.vb(data = stan_data, 
                                iter = iter_cnt,
                                output_samples = output_cnt)
        
            para_dic = {}
            for j in range(0, len(fit['sampler_param_names'])):
                para_dic[fit['sampler_param_names'][j]] = fit['sampler_params'][j]
            customer_para[i] = para_dic
        except:
            print(i + ' will not output result.')
            ccc.append(i)
######################################################################  
        # cnt1 = cnt1 + 1
        # if cnt1 == 20:
        #     break
######################################################################   
    purchase_info_dic = {'customer': customer_name_list,
                          'purchase_cnt': purchase_cnt}
    
    return customer_para, purchase_info_dic

def create_info_dic(select_material, spec_last, deal_last, dic):
    now = datetime.datetime.now()
    
    stan_info_dic = {'Date': str(now.year)+'/'+str(now.month)+'/'+str(now.day),
                         'Duration for Spec Data': spec_last,
                         'Duration for Deal Data': deal_last,
                          'Selected Material': select_material,
                          'Input Variable': '\n'.join([i +' : '+ ', '.join(dic[i]) for i in dic.keys()])}
    return stan_info_dic

def name_next_stan_file(mypath):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    subs = '_stan_result'
    subs2 = '.json'
    files = [i for i in files if subs in i]
    model_num = [int(i.split(subs)[1].split(subs2)[0]) for i in files]
    
    if model_num == []:
        count = 0
    else:
        count = max(model_num)
        
    count = count + 1
    return str(count)

def find_usable_file(mypath):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    subs = '_info'
    files = [i for i in files if subs in i]
    
    dic_list = []
    for i in range(len(files)):
        with open(files[i], 'r') as j: 
            info_dic = json.load(j)
        dic_list.append(info_dic)
    
    return dic_list

def delete_and_rename_file(mypath, delete_value, recover_value,
                           choose_value, select_material):
    sub1 = select_material + '_info'
    sub2 = select_material + '_stan_result'
    sub3 = '.json'
    sub4 = select_material + '_history'
    sub5 = '.csv'
    
    reserve_file = []

    for i in range(len(delete_value)):
        if (delete_value[i] - recover_value[i]) > 0 and choose_value[i] == 0:
            try:
                print(mypath + '/' + sub1 + str(i+1) + sub3)
                os.remove(mypath + '/' + sub1 + str(i+1) + sub3)
            except:
                print(sub1 + str(i+1) + sub3 + ''' doesn't exist!''')
            try:
                os.remove(mypath + '/' + sub2 + str(i+1) + sub3)
            except:
                print(sub2 + str(i+1) + sub3 + ''' doesn't exist!''')
            try:
                os.remove(mypath + '/' + sub4 + str(i+1) + sub5)
            except:
                print(sub4 + str(i+1) + sub5 + ''' doesn't exist!''')
        else:
            reserve_file.append(i)
    
    for j in range(len(reserve_file)):
        os.rename(mypath + '/' + sub1 + str(reserve_file[j]+1) + sub3,
                  mypath + '/' + sub1 + str(j+1) + sub3)
        os.rename(mypath + '/' + sub2 + str(reserve_file[j]+1) + sub3,
                  mypath + '/' + sub2 + str(j+1) + sub3)
        os.rename(mypath + '/' + sub4 + str(reserve_file[j]+1) + sub5,
                  mypath + '/' + sub4 + str(j+1) + sub5)
    
    return True


def dic_string_to_list(str):
    substr = str.split('\n')
    focus = []
    reserve = []
    
    for i in range(len(substr)):
        key = substr[i].split(' : ')[0]
        other = substr[i].split(' : ')[1].split(', ')
        focus = focus + [key]
        if isinstance(other, list):
            reserve = reserve + [key] + other
        else:
            reserve = reserve + [key] + [other]
        
    return {'input': reserve, 'focus': focus}


def return_table_variable(dic1, dic2, material):
    
    if dic1 is None and dic2 is None:
        return {}
    elif dic1 is None:        
        if dic2['Selected Material'] == material:
            dic = dic2
        else:
            return {}
    elif dic2 is None:
        if dic1['Selected Material'] == material:
            dic = dic1
        else:
            return {}
    else:
        if dic1['Selected Material'] == material and dic2['Selected Material'] == material:
            if dic1['activate time'] > dic2['activate time']:
                dic = dic1
            else:
                dic = dic2
        elif dic1['Selected Material'] == material:
            dic = dic1
        elif dic2['Selected Material'] == material:
            dic = dic2
        else:
            return {}
            
    return_dic = dic_string_to_list(dic['Input Variable'])
    return_dic['Mean and Var'] = dic['Mean and Var']
    return_dic['purchase_info'] = dic['purchase_info']
    return return_dic
                 

    
def rating_system(input_spec_value, # the spec information of current product
                  focus_spec_name,
                  customer_para,
                  reserve_spec,
                  mean_and_sd,
                  ranking_method):
    
    
    rating_df = pd.DataFrame(columns=['customer id'] + [i+j for i in focus_spec_name for j in ['_mean', '_sigma']] + [i+'_score' for i in focus_spec_name])
    score_df = pd.DataFrame(columns=['customer id', 'score'])
    
    mean_vec = mean_and_sd['mean']
    sd_vec = mean_and_sd['sd']
    input_value_sd = [(input_spec_value[i] - mean_vec[i]) / sd_vec[i] 
                     for i in range(0, len(sd_vec))]     

    customer_No = list(customer_para.keys())
    customer_cnt = 0
    for i in customer_No:
        store_value = []
        score_value = []
        for j in focus_spec_name:
            spec_index = reserve_spec.index(j)
            current_spec_index = 0
            create_cnt = 0
            sd_val_store = []
            for k in range(0, len(input_spec_value)):
                if k == spec_index:
                    continue
                
                current_spec_index = current_spec_index + 1
                # input_value_sd = (input_spec_value[k] - mean_vec[k]) / sd_vec[k]
                # sd_val_store.append(input_value_sd)
                
                if create_cnt == 0:
                    mean_array = np.array(customer_para[i]['A['+str(spec_index+1)+','+str(current_spec_index)+']'])*input_value_sd[k]
                else:
                    mean_array = mean_array + np.array(customer_para[i]['A['+str(spec_index+1)+','+str(current_spec_index)+']'])*input_value_sd[k]
                create_cnt = create_cnt + 1
            
            mean_array = mean_array + np.array(customer_para[i]['A_mu['+str(spec_index+1)+']'])
            mean_value = np.mean(mean_array)
            sigma_value = np.mean(np.array(customer_para[i]['sigma['+str(spec_index+1)+']']))
            
            store_value.append(mean_value)
            store_value.append(sigma_value)
            if ranking_method == 1:
                score_value.append((input_value_sd[spec_index] - mean_value)/sigma_value)
            else:
                score_value.append(input_value_sd[spec_index] - mean_value)
            
        rating_df.loc[customer_cnt] = [i] + store_value + score_value
        customer_cnt = customer_cnt + 1
     
    score_df['customer id'] = rating_df['customer id']
    score_array = np.array([0 for i in range(score_df.shape[0])])
    for i in focus_spec_name:
        score_array = np.power(np.asarray(rating_df[i+'_score']), 2) + score_array
    score_array = np.sqrt(score_array)
    score_df['score'] = score_array

    return rating_df, score_df

# rate, score = rating_system([1,2,3,4,5,6,7,8],
#                             ['色相(調)  -  L','Film - 薄片污點(2)'],stan_result,
#                             dic['input'], dic['Mean and Var'])
    
def cust_to_class(cut_off, purchase_info):
    class_cnt = len(cut_off) + 1
    
    if class_cnt == 2:
        class_name = ['Large', 'Small']
    elif class_cnt == 3:
        class_name = ['Large', 'Median', 'Small']
    elif class_cnt == 1:
        class_name = ['Total']
    else:
        class_name = ['First', 'Second', 'Third', 'Fourth', 'Fifth',
                      'Sixth', 'Seventh']
        
    purchase_cnt = purchase_info['purchase_cnt']
    name_list = purchase_info['customer']
    
    cut_off = [0] + cut_off + [1]
    cut_off.reverse()

    quan_cut_off = [round( np.quantile(purchase_cnt, cut_off[i]) ) for i in range(class_cnt+1)]
    # two different quantile should not have same cut point
    for k in range(len(quan_cut_off)-1, 1, -1):
        if quan_cut_off[k] >= quan_cut_off[k-1]:
            quan_cut_off[k-1] = quan_cut_off[k] + 1

    purchase_cnt = np.array(purchase_cnt)
    cust_class_dic = {}
    for i in range(class_cnt):
            
        if i < class_cnt-1:
            bool_vec = np.logical_and(purchase_cnt > quan_cut_off[i+1],
                                      purchase_cnt <= quan_cut_off[i])
        else:
            bool_vec = np.logical_and(purchase_cnt >= quan_cut_off[i+1],
                                      purchase_cnt <= quan_cut_off[i]) 

        cust_class_dic[class_name[i]] = list(compress(name_list, list(bool_vec)))
    
    return cust_class_dic, quan_cut_off
  
      
def cust_rating(cust_dic, score, choose_cnt, type, customer):
    cust_rating_dic = {}
    score_dic = {}
   
    if type:
        reserve = []
        for i in cust_dic.keys(): reserve = reserve + cust_dic[i]
        cust_dic = {'ALL': reserve}
    
    all_no = [str(j) for j in customer['客戶編號']]
    all_name = list(customer['名稱'])
    
    flexible_cnt = False
    if isinstance(choose_cnt, str):
        flexible_cnt = True
        
    for i in cust_dic.keys():
        bool_vec = [j in cust_dic[i] for j in list(score['customer id'])]
        class_df = score.loc[bool_vec,:]
        class_df.sort_values('score', axis=0, ascending=True, inplace=True)
        
        max_len = class_df.shape[0]
        if flexible_cnt:
            choose_cnt = max_len
            
        if choose_cnt <= max_len:
            chosen_no = list(class_df['customer id'])[0: choose_cnt]
            chosen_score = list(class_df['score'])[0: choose_cnt] 
        else:
            chosen_no = list(class_df['customer id'])[0: max_len] + [
                'NAN' for j in range(choose_cnt - max_len)]
            chosen_score = list(class_df['score'])[0: max_len] + [
                'NAN' for j in range(choose_cnt - max_len)]
        
        # to real name
        reserve_name = []
        for k,j in enumerate(chosen_no):
            try:
                index_record = all_no.index(j)
                reserve_name.append(all_name[index_record])
            except:
                reserve_name.append(chosen_no[k])
     
        cust_rating_dic[i] = reserve_name
        score_dic[i] = chosen_score
            
    return cust_rating_dic, score_dic


def score_name_congugate(name, score):
    congugate_dic = {}
    for i in name.keys():
        new_vec = []
        for j,k in enumerate(name[i]):
            if not isinstance(score[i][j], str):
                new_vec.append(k + ' (' + str( round(score[i][j],2) ) + ')')
            else:
                new_vec.append(k + ' (NAN)')
        congugate_dic[i] = new_vec
    
    return congugate_dic
    
    

def check_table_diff(cur_data, pre_data):
    """
    Check which value of spec in the table "input_spec_inform" is modified.
    """

    cur_original_data = cur_data[0]
    cur_standardized_data = cur_data[1]

    pre_original_data = pre_data[0]
    pre_standardized_data = pre_data[1]

    modified_data = None
    modified_spec = None

    for spec, cur_val in cur_original_data.items():
        if spec != 'Format':
            if pre_original_data[spec] != cur_val:
                modified_data = 'Original'
                modified_spec = spec

    for spec, cur_val in cur_standardized_data.items():
        if spec != 'Format':
            if pre_standardized_data[spec] != cur_val:
                modified_data = 'Standardized'
                modified_spec = spec

    return modified_data, modified_spec


def customer_name_to_id(cus_df, cus_name):
    name_list = list(cus_df['名稱'])
    try:
        # 2 is column index for customer id
        cus_id = cus_df.iloc[name_list.index(cus_name), 2]
        cus_id = str(cus_id)
    except:
        cus_id = cus_name
        
    return cus_id
            

            
            
            
            
            
            
            