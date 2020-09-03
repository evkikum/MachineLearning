#Class work: Calculate Moving Avg, Roll std and exponentially weighted moving average and plot all 3 in one graph

#simple plot with all above the data
plt.plot(ts)
plt.plot(moving_avg, color='red')
plt.plot(moving_std, color='blue')
plt.plot(moving_ewma, color='green')
plt.show()
