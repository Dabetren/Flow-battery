#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime
from numpy import trapz
from datetime import datetime
name_file = "e13_cycling_100mAcm2.txt"  #файл с данными 
f_incoming_data = open(name_file)  # открываем файл с данными 
header = ['Cycle', 'Flag', 'T_sum', 'S', 'V', 'A']  # прописываем заголовок для нового файла 
dic_area = {}  # словарь для записи ключ - значение (название цикла и подцикла : результат интегрирования)
cycle_num = 0  # переменная для обозначения цикла 
cd_flag = 0  # переменная для обозначения подцикла заряда (1) и разряда (2)
T_sum = 0  # переменная для суммарного времени работы
n = (1, 2)  # список подциклов 1 - заряд, 2 - разряд 
current_datetime = datetime.now().strftime('%d-%m-%Y_%H-%M')  # переменная с датой и временем 
with open(f"new_{name_file[ :-4]}_{current_datetime}.csv", 'w', newline = '') as f_results_table:  # открываем файл для записи обработанных данных и обозначаем в переменную
    writer=csv.writer(f_results_table, delimiter = ';')  # заводим переменную для записи в файл csv с разделителем ';'
    writer.writerow(header)  # записываем в файл переменную (header) с заголовком 
    for line in f_incoming_data:  # проходимся по каждой линии в исходном файле 
        if (not line.startswith(tuple('0123456789')) or not line.strip()):  # если линия не начинается с цифры или пробела
            if line.startswith('Cycle'):  # если линия начинается с слова 'Cycle'
                cycle_num = int(line.partition(' ')[-1])  # записываем в переменную последнее знак линии (цифру) 
            elif line.startswith("Step 1"):  # если линия начинается с слова 'Step 1'
                cd_flag = 1  # заносим в переменную (1) т.е. заряд
            elif line.startswith("Step 2"):  # если линия начинается с слова 'Step 2'
                cd_flag = 2  # заносим в переменную (2) т.е. разряд 
        else:  
            if  T_sum == 0:  # определяем первая ли это итерация записи в файл, если первая, то
                Time = (float(line[0 : 5]))  # находим шаг времени для всего файла 
                T_sum = T_sum + Time  # начинаем считать суммарное время для всего файла
                cycle_flag_T_sum = [cycle_num, cd_flag, T_sum]  # объединяем значения цикла, подцикла заряда/разряда и значение суммарного времени в список
                line = cycle_flag_T_sum + line.split()  # складываем список 'cycle_flag_T_sum' и значения строки 'line' поделенной по пробелам
                writer = csv.writer(f_results_table, delimiter = ';')  # заводим переменную для записи в файл csv с разделителем ';'
                writer.writerow(line)  # записываем подготовленную строку в файл
            else:
                T_sum = T_sum + Time  # если не первая итерация записи, то продолжваем считать суммарное время
                cycle_flag_T_sum = [cycle_num, cd_flag, T_sum]  # объединяем значения цикла и заряда/разряда в список
                line = cycle_flag_T_sum + line.split()  # складываем список 'cycle_flag__T_sum' и значения строки 'line' поделенной по пробелам
                writer = csv.writer(f_results_table, delimiter = ';')  # заводим переменную для записи в файл csv с разделителем ';'
                writer.writerow(line)   # записываем подготовленную строку в файл

pf = pd.read_csv(f"new_{name_file[ :-4]}_{current_datetime}.csv", sep = ';')  # создаем dataframe из подготовленного файла в pandas, где делителем служит ':'
pf_new = pf.groupby(['Cycle', 'Flag'], as_index = False).aggregate({'A' : 'sum', 'S' : 'max', "T_sum" : 'max'})  # создаем новый датафрейм на основе колонок первого 'Cycle' и 'Flag' и применяем математические действия для колонок T(берем макс значение), T_sum(берем макс значение) и A(берем сумму)  
T_sum = pf_new['S'].sum()   # находим максимальное значение времени для всех данных 
Cycle_ex = pf['Cycle'].max()  # находим количество циклов 

for i in range(1, Cycle_ex + 1, 1):  # перобор циклов от первого до мах 
    for m in n:  # перебор подциклов заряда и разряда 
        y = pf[(pf.Cycle == i)&(pf.Flag == m)]['A']  # переменная содержашая в себе значения тока для определенного цикла и подцикла 
        area = trapz(y, dx = Time)  #  интегрирование тока по времени для каждого цикла и подцикла, где dx шаг времени 
        dic_area[f"{i}_{m}"] = area  # заполняем словарь, ключ - цикл_подцикл : значение - заряд
        
pf_area = pd.DataFrame(dic_area.items(), columns = ["Cycle_Flag", 'Charge'])  # датафрейм на основе полученного словаря
pf_area = pf_area[pf_area.Charge != 0]  # удаление лишней строки, когда заряд равен 0
pf_area[['Cycle', 'Flag']] = pf_area.Cycle_Flag.str.split('_', expand = True).apply(pd.to_numeric)  # делим столбец отвечающий за цикл_подцикл на два и переводим значения полученных столбцов в формат int 
pf_area_new = pf_area.groupby(['Cycle'], as_index = False).Charge.agg([
    lambda x : x.max(), 
    lambda x : x.min()
    ])  # из датафрейма pf_area по столбцу 'Cycle' делим полученные пары значений для подциклов разряда(минимальное знач) и заряда(максимальное знач) на два отдельных столбца
pf_area_new = pf_area_new.rename(columns = {"<lambda_0>" : "Q_charge", "<lambda_1>" : "Q_recharge"})  # переименовываем колонки из вида <lambda_х> в читаемый вид
pf_area_new["current_efficiency"] = abs(pf_area_new["Q_recharge"] / pf_area_new["Q_charge"])  # находим current efficiency

pf_voltaic = pf.groupby(['Cycle', 'Flag'], as_index = False).aggregate({'V' : 'mean'})  # создаем новый датафрейм на основе первого, на основе колонок 'Cycle' и 'Flag' и применяем математические действия для колонки V(берем среднее значение)
pf_voltaic_efficiency = pf_voltaic.groupby(['Cycle'], as_index = False).V.agg([
    lambda x : x.max(), 
    lambda x : x.min()
    ])  # из датафрейма pf_voltaic по столбцу 'Cycle' делим полученные пары значений для подциклов разряда(минимальное знач) и заряда(максимальное знач) на два отдельных столбца 
pf_voltaic_efficiency = pf_voltaic_efficiency.rename(columns={"<lambda_0>" : "V_charge", "<lambda_1>" : "V_recharge"})  # переименовываем колонки из вида <lambda_х> в читаемый вид
pf_voltaic_efficiency["voltaic_efficiency"] = abs(pf_voltaic_efficiency["V_recharge"] / pf_voltaic_efficiency["V_charge"])  # находим voltaic efficiency

pf_mean_A = abs(pf['A']).mean()  # среднее значение Тока для всего файла 
pf_voltaic_efficiency['R'] = ((pf_voltaic_efficiency["V_charge"] - pf_voltaic_efficiency["V_recharge"]) / (2 * pf_mean_A))  # находим voltaic efficiency

pf_cycle_max_time = pf_new.groupby(['Cycle'], as_index = True).aggregate({"T_sum" : 'max'})  # создаем новый датафрейм, по циклам и находим максимальное время дла каждого 
pf_cycle_max_time['R'] = pf_voltaic_efficiency['R']  # добавляем в него значения сопротивления  из другого датафрейма 
pf_cycle_max_time = pf_cycle_max_time[pf_cycle_max_time.R != 0]  # убираем некоректные значения сопротивления, когда R = 0

result = pd.concat([pf_area_new["current_efficiency"], pf_voltaic_efficiency["voltaic_efficiency"]], axis = 1, join = 'inner')  # соединяем нужные колонки с КПД из д датафреймов в один по оси x(axis=1), так что было пересечение датафреймов(join='inner')
result["energy_efficiency"] = result["current_efficiency"] * result["voltaic_efficiency"]  # находим energy efficiency
result[["Q_charge", "Q_recharge"]] = pf_area_new[["Q_charge", "Q_recharge"]]  # добавляем в датафрейм значения Q зарядки/разрядки 
result[["V_charge", "V_recharge", 'R']] = pf_voltaic_efficiency[["V_charge", "V_recharge", 'R']]  # добавляем в датафрейм значения V зарядки/разрядки и R
result = result[result.R != 0]  # убираем некоректные данные сопротивления, когда R = 0
result.to_csv(f"result_efficiency_{name_file[ :-4]}_{current_datetime}.csv", sep=';')  #сохраняем таблицу со всеми КПД  и значениями в отдельный файл
result

# # строим график AT

fig = plt.figure(figsize = (30, 15))  # создаем объект Figure c заданным размером в дюймах
x = pf["T_sum"]  # присваиваем значения x
y = pf['A']  # присваиваем значения 
myhex = '#0086cb'  # цвет по HEX
AT = fig.add_subplot(111)  # добавляем область рисования
AT.plot(x, y, color = myhex, linewidth = 1.5, linestyle = 'solid')  # выводим график, определяем цвет линии, толщину, стиль
AT.grid(True, linewidth = 1, linestyle = 'solid')  # линии вспомогательной сетки, ее тоцщина и стиль
xAT = AT.xaxis  # обращаемся к оси х
yAT = AT.yaxis  # обращаемся к оси у
xlabels = xAT.get_ticklabels()
ylabels = yAT.get_ticklabels()
for label in xlabels:
    label.set_color('black')  # цвет подписи деленений оси x
    label.set_fontsize(14)  # размер шрифта подписей делений оси x 
for label in ylabels:
    label.set_color('black')  # цвет подписи деленений оси y
    label.set_fontsize(14)  # размер шрифта подписей делений оси y 
xticks = AT.get_xticks()  # заносим в переменную местоположения делений по х
yticks = AT.get_yticks()  # заносим в переменную местоположения делений по y 
# # xx = np.arange(0, T_sum, 10000)  # выбираем нужный диапазон делений по х с необходимым шагом
# # yy = np.arange(-0.5, 0.5, 0.1)  # выбираем нужный диапазон делений по y с необходимым шагом
# # AT.set_xticks(xx)  # задаем значения оси x
# # AT.set_yticks(yy)  # задаем значения оси y
fig.savefig(f"graph_AT_{name_file[ :-4]}_{current_datetime}.png", dpi = 1000, bbox_inches = 'tight', facecolor = 'white')  # сохраняем график с выбором DPI(разрешения),плотной обрезки по осям и отоброжении осей  \

# # строим график VT

fig = plt.figure(figsize = (15, 15))  # создаем объект Figure c заданным размером в дюймах
x = pf["T_sum"]  # присваиваем жначения x
y = pf['V']  # присваиваем жначения 
myhex = '#0086cb'  # цвет по HEX
VT = fig.add_subplot(111)  # добавляем область рисования
VT.plot(x, y, color = myhex, linewidth = 1.5, linestyle = 'solid')  # выводим график, определяем цвет линии, толщину, стиль
VT.grid(True, linewidth = 1, linestyle = 'solid')  # линии вспомогательной сетки, ее тоцщина и стиль
xVT = VT.xaxis  # обращаемся к оси х
yVT = VT.yaxis  # обращаемся к оси у
xlabels = xVT.get_ticklabels()
ylabels = yVT.get_ticklabels()
for label in xlabels:
    label.set_color('black')  # цвет подписи деленений оси x
    label.set_fontsize(14)  # размер шрифта подписей делений оси x 
for label in ylabels:
    label.set_color('black') # цвет подписи деленений оси y
    label.set_fontsize(14) # размер шрифта подписей делений оси y 
xticks = VT.get_xticks()  # заносим в переменную местоположения делений по х
yticks = VT.get_yticks()  # заносим в переменную местоположения делений по y
# # xx = np.arange(0, 195000, 15000)  # выбираем нужный диапазон делений по х с необходимым шагом
# # yy = np.arange(0.7, 1.7, 0.1)  # выбираем нужный диапазон делений по y с необходимым шагом
# # VT.set_xticks(xx)  # задаем значения оси x
# # VT.set_yticks(yy)  # задаем значения оси y
fig.savefig(f"graph_AV_{name_file[ :-4]}_{current_datetime}.png", dpi = 1000, bbox_inches = 'tight', facecolor = 'white')  # сохраняем график с выбором DPI(разрешения),плотной обрезки по осям и отоброжении осей  

# # cтроим график QT

fig = plt.figure(figsize = (30, 15))  # создаем объект Figure c заданным размером в дюймах
x = pf_new["T_sum"]  # присваиваем значения x
y = pf_area['Charge']  # Присваиваем значения 
myhex = '#0086cb'  # цвет по HEX
QT = fig.add_subplot(111)  # добавляем область рисования
QT.plot(x, y, color = myhex, linewidth = 1.5, linestyle = 'solid')  # выводим график, определяем цвет линии, толщину, стиль
QT.grid(True, linewidth = 1, linestyle = 'solid')  # линии вспомогательной сетки, ее тоцщина и стиль
xQT = QT.xaxis  # обращаемся к оси х
yQT = QT.yaxis  # обращаемся к оси у
xlabels = xQT.get_ticklabels()
ylabels = yQT.get_ticklabels()
for label in xlabels:
    label.set_color('black')  # цвет подписи деленений оси x
    label.set_fontsize(14)  # размер шрифта подписей делений оси x 
for label in ylabels:
    label.set_color('black')  # цвет подписи деленений оси y
    label.set_fontsize(14)  # размер шрифта подписей делений оси y 
xticks = QT.get_xticks()  # заносим в переменную местоположения делений по х
yticks = QT.get_yticks()  # заносим в переменную местоположения делений по y 
# # xx = np.arange(0, T_sum, 10000)  # выбираем нужный диапазон делений по х с необходимым шагом
# # yy = np.arange(-0.5, 0.5, 0.1)  # выбираем нужный диапазон делений по y с необходимым шагом
# # QT.set_xticks(xx)  # задаем значения оси x
# # QT.set_yticks(yy)  # задаем значения оси y
fig.savefig(f"graph_QT_{name_file[ :-4]}_{current_datetime}.png", dpi = 1000, bbox_inches = 'tight', facecolor = 'white')  # сохраняем график с выбором DPI(разрешения),плотной обрезки по осям и отоброжении осей  \

# # cтроим график RT

fig = plt.figure(figsize = (30, 15)) # создаем объект Figure c заданным размером в дюймах
x = pf_cycle_max_time["T_sum"]  # присваиваем значения x
y = pf_cycle_max_time['R']  # присваиваем значения 
myhex = '#0086cb'  # цвет по HEX
RT = fig.add_subplot(111)  # добавляем область рисования
RT.plot(x, y, color = myhex, linewidth = 1.5, linestyle = 'solid')  # выводим график, определяем цвет линии, толщину, стиль
RT.grid(True, linewidth = 1, linestyle = 'solid')  # линии вспомогательной сетки, ее тоцщина и стиль
xRT = RT.xaxis  # обращаемся к оси х
yRT = RT.yaxis  # обращаемся к оси у
xlabels = xRT.get_ticklabels()
ylabels = yRT.get_ticklabels()
for label in xlabels:
    label.set_color('black')  # цвет подписи деленений оси x
    label.set_fontsize(14)  # размер шрифта подписей делений оси x 
for label in ylabels:
    label.set_color('black')  # цвет подписи деленений оси y
    label.set_fontsize(14)  # размер шрифта подписей делений оси y 
xticks = RT.get_xticks()  # заносим в переменную местоположения делений по х
yticks = RT.get_yticks()  # заносим в переменную местоположения делений по y 
# # xx = np.arange(0, T_sum, 10000)  # выбираем нужный диапазон делений по х с необходимым шагом
# # yy = np.arange(0.7, 0.8, 0.1)  # выбираем нужный диапазон делений по y с необходимым шагом
# # RT.set_xticks(xx)  # задаем значения оси x
# # RT.set_yticks(yy)  # задаем значения оси y
fig.savefig(f"graph_RT_{name_file[ :-4]}_{current_datetime}.png", dpi = 1000, bbox_inches = 'tight', facecolor='white')  # сохраняем график с выбором DPI(разрешения),плотной обрезки по осям 


# In[ ]:




