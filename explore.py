import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wrangle as w
import model as m
import os
import seaborn as sns
from math import sqrt
from scipy import stats
from pydataset import data
# ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_tl_boy(train):
    tall_boy= train[train.height > 75]
    large_boy= train[train.weight > 280]
    tl_boy= pd.concat([tall_boy, large_boy])
    tl_wis= tl_boy.wisdom
    
    overall= train.wisdom.mean()
    return tl_boy, tl_wis, overall

def plot_above(train, tl_boy):
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})

    fig = plt.figure()
    ax = fig.add_subplot(221)
    sns.countplot(data= train, x= 'wisdom')

    ax.set_xlabel('Wisdom Sta', fontsize = 12)
    ax.set_ylabel('Character Count', fontsize = 12)
    ax.set_title('All Stats', fontsize = 12)

    ax = fig.add_subplot(222)
    sns.countplot(data= tl_boy, x= 'wisdom')
    ax.set_xlabel('Wisdom Stat', fontsize = 12)
    ax.set_ylabel('Character Count', fontsize = 12)
    ax.set_title('Above Height & Weight Stat', fontsize = 12)
    plt.show()



def the_t(train, tl_wis):
    alpha = 0.5
    overall= train.wisdom.mean()
    
    t, p = stats.ttest_1samp(tl_wis, overall)

    if p/2 > alpha:
        print("We fail to reject the Null Hypothesis")
    elif t < 0:
        print("We fail to reject the Null Hypothesis")
    else:
        print("We reject the Null Hypothesis")


def get_above(train):
    
    above_height= train[train.height > 60]
    above_weight= train[train.weight > 145]
    above_speed= train[train.speed > 28]
    above_strength= train[train.strength > 13]
    above_dex= train[train.dexterity > 13]
    above_constitution= train[train.constitution > 13]
    above_intelligence= train[train.intelligence > 13]
    above_charisma= train[train.charisma > 13]
    
    above_df= pd.concat([above_charisma, above_constitution, above_dex, above_height, above_intelligence,
                   above_speed, above_weight, above_strength])
    
    above_wis= above_df.wisdom
    
    overall= train.wisdom.mean()
    return above_df, above_wis, overall


def plot_above(train, above_df):
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})

    fig = plt.figure()
    ax = fig.add_subplot(221)
    sns.countplot(data= train, x= 'wisdom')

    ax.set_xlabel('Wisdom Stat', fontsize = 12)
    ax.set_ylabel('Character Count', fontsize = 12)
    ax.set_title('All Stats', fontsize = 12)

    ax = fig.add_subplot(222)
    sns.countplot(data= above_df, x= 'wisdom')
    ax.set_xlabel('Wisdom Stat', fontsize = 12)
    ax.set_ylabel('Character Count', fontsize = 12)
    ax.set_title('Above Average Stats', fontsize = 12)
    plt.show()


def test_above(above_wis, overall):
    alpha= 0.5
    t, p = stats.ttest_1samp(above_wis, overall)

    if p/2 > alpha:
        print("We fail to reject the Null Hypothesis")
    elif t < 0:
        print("We fail to reject the Null Hypothesis")
    else:
        print("We reject the Null Hypothesis")


def get_big(train):    
    big_smart= train[train.intelligence > 16]
    big_brain= big_smart[big_smart.dexterity > 16]
    return big_brain


def plot_big_brain(train, big_brain):
    '''
    This function is used to plot show the wisdom difference between all characters, 
    and those who have both high intelligence and dexterity.
    '''
    fig = plt.figure()
    sns.set(font_scale = 3)
    ax1 = fig.add_subplot(223)
    sns.set(rc={'figure.figsize':(30.7,24.27)})
    sns.set(font_scale = 3)
    sns.countplot(data= train, x= 'wisdom')

    ax1.set_xlabel('Wisdom Stat', fontsize = 35)
    ax1.set_ylabel('Character Count', fontsize = 35)

    ax2 = fig.add_subplot(224)
    sns.set(rc={'figure.figsize':(30.7,24.27)})
    sns.countplot(data= big_brain, x= 'wisdom')
    ax2.set_xlabel('Wisdom Stat', fontsize = 35)
    ax2.set_ylabel('Character Count', fontsize = 35)
    ax2.set_title('High Intelligence/Dexterity', fontsize = 35)
    plt.show()


def get_avg(train):
    
    '''
    This function is used to get the average of each character stat.
    '''
    dragonborn_wis= train[train.race== 'dragonborn'].wisdom.mean()
    halfling_wis= train[train.race== 'halfling'].wisdom.mean()
    gnome_wis= train[train.race== 'gnome'].wisdom.mean()
    human_wis= train[train.race== 'human'].wisdom.mean()
    tiefling_wis= train[train.race== 'tiefling'].wisdom.mean()
    elf_wis= train[train.race== 'elf'].wisdom.mean()
    dwarf_wis= train[train.race== 'dwarf'].wisdom.mean()
    half_elf_wis= train[train.race== 'half.elf'].wisdom.mean()
    half_orc= train[train.race== 'half.orc'].wisdom.mean()
    wis_avg= pd.DataFrame({'dragonborn': dragonborn_wis,
                  'halfling': halfling_wis,
                  'gnome': gnome_wis,
                  'human': human_wis,
                  'tiefling': tiefling_wis,
                  'elf': elf_wis,
                  'dwarf': dwarf_wis,
                  'half_elf': half_elf_wis,
                  'half_orc': half_orc}, index= [0])
    wis_mean= pd.DataFrame(train.groupby('race')['wisdom'].mean()).reset_index()
    return wis_mean


def wis_per_race(wis_mean):
    
    '''
    This function is used to plot each race and their respective wisdom average.
    '''
    color= ['grey', 'grey', 'grey', 'grey', 'grey', 'grey', 'grey','firebrick']
    sns.set(rc={'figure.figsize':(15.7, 9.27)})
    sns.set(font_scale = 1.5)
    fig = plt.figure()
    sns.barplot(data= wis_mean, y= 'wisdom', x= 'race', palette= color, ec= 'black')
    sns.set(rc={'figure.figsize':(11.0, 7.50)})
    plt.xlabel('Character Race Type', fontsize = 20)
    plt.ylabel('Wisdom Stat Count', fontsize = 20)
    plt.title('Wisdom Stat Per Race Type', fontsize = 20)

    plt.show()