#The original data were traced to the Colliery Year Book and Coal Trades Directory, available
#from the National Coal Board in London. Publication ceased in 1962, but issues before
#then give data on explosions in coal mines back to 15 March 1851. Accordingly, the data have
#been corrected and extended to cover the period 15 March 1851 to 22 March 1962 inclusive,
#a total of 40,550 days. There were 191 explosions involving 10 or more men killed, including
# Raftery and Akman also note that no explosions occurred Jan1 1951 to March 15 1851 nor between 23 March 1962 to December 31 1961, so we can add to the first and last numbers of Jarrett
# we add 73 to the first one
# we add 273 to the last one
import numpy as np
from matplotlib import pyplot as plt

def get_x():
    x = np.array([[157+73, 65, 53, 93, 127, 176, 22, 1205, 1643, 312], 
    [123, 186, 17, 24, 218, 55, 61, 644, 54, 536], 
    [2, 23, 538, 91, 2, 93, 78, 467, 326, 145], 
    [124, 92, 187, 143, 0, 59, 99, 871, 1312, 75], 
    [12, 197, 34, 16, 378, 315, 326, 48, 348, 364], 
    [4, 431, 101, 27, 36, 59, 275, 123, 745, 37], 
    [10, 16, 41, 144, 15, 61, 54, 456, 217, 19], 
    [216, 154, 139, 45, 31, 1, 217, 498, 120, 156], 
    [80, 95, 42, 6, 215, 13, 113, 49, 275, 47], 
    [12, 25, 1, 208, 11, 189, 32, 131, 20, 129], 
    [33, 19, 250, 29, 137, 345, 388, 182, 66, 1630], 
    [66, 78, 80, 112, 4, 20, 151, 255, 292, 29], 
    [232, 202, 3, 43, 15, 81, 361, 194, 4, 217], 
    [826, 36, 324, 193, 72, 286, 312, 224, 368, 7], 
    [40, 110, 56, 134, 96, 114, 354, 566, 307, 18], 
    [12, 276, 31, 420, 124, 108, 307, 462, 336, 1358], 
    [29, 16, 96, 95, 50, 188, 275, 228, 19, 2366], 
    [190, 88, 70, 125, 120, 233, 78, 806, 329, 952], 
    [97, 225, 41, 34, 203, 28, 17, 517, 330, 632 + 273]])
    x = x.reshape(x.size, order='F')
    return x

def get_accident_days():
    x = get_x()

    num_disasters = x.size

    cumulative_days = np.cumsum(x)
    return cumulative_days

def plot_cumulative_days(cumulative_days):
    day_grid = np.linspace(1, cumulative_days[-1], cumulative_days[-1])
    disaster_vals = np.zeros(day_grid.size)

    num_disasters = 0
    i = 0
    while i < day_grid.size:
        if i < cumulative_days[num_disasters]:
            disaster_vals[i] = num_disasters
            i += 1
        else:
            num_disasters += 1



    #plt.plot(cumulative_days/ 1000, np.linspace(1, num_disasters, num_disasters), 'k')
    plt.plot(day_grid/1000, disaster_vals, 'k')

    plt.yticks([i*25 for i in range(9)])
    plt.xticks([i*4 for i in range(11)])
    plt.ylim([0, 200])
    plt.xlim([0, 40])
    plt.show()

def plot_day_histogram():
    x = get_x()
    import datetime as dt
    # start date is Jan 1 1851
    start_date = dt.datetime(1851, 1, 1)
    date_list = [start_date]
    curr_date = start_date

    print(dt.timedelta(days=1))



    for i in range(x.size):
        print(x[i])
        delta = dt.timedelta(days=int(x[i]))
        curr_date = curr_date + delta
        date_list.append(curr_date)

    years = [d.year for d in date_list]

    all_years = np.linspace(1851, 1962, 1962-1851+1)
    year_accidents = np.zeros(len(all_years))
    ind = 0
    for year in all_years:
        count = 0
        while ind < len(years) and years[ind] == year:
            count += 1
            ind += 1
        year_accidents[int(year-1851)] = count
    plt.figure()
    plt.plot(all_years, year_accidents)
    plt.show()


    print(years)
    print(start_date)

if __name__ == '__main__':

    cumulative_days = get_accident_days()
    plot_cumulative_days(cumulative_days)
    plot_day_histogram()

    



