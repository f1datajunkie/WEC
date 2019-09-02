# ## Pit Analysis
#
# We can also start to analyse pit performance more generally.
#
# Trivially, we might calculate the number of pit stops and the accumulated pit time, perhaps subtracting this from the accumulated race time.
#
# We might also compare laptimes on the lap immediately before the in-lap and immediately after the out-lap.

# ## Base Analysis
#
# A basic analysis simply reports on the number of pit events associated with each car and driver.
#
# To get a feel for when pit events occur for each car, we might start out with a simple heatmap display:



# We can also generate a simple count of pit events and accumulated pit stop time for each car, displaying the result as a simple stacked bar chart (for comparing accumulated stop times) and  dodged bar chart for comparing pit times associated with each driver within each team.
#
# We might also want to differentiate performance associated with pit stops that include a driver change from pit stops managed by a single driver.

# ## Exploring Pit Event Losses and Gains
#
# If we pit on lap $n$ with laptime $l_n$ then we can start to pick apart lap times associated with pit events:
#
# - *inlap*: $l_n$, the laptime for the inlap;
# - *outlap*: $l_{n+1}$, the laptime for the outlap;
# - *pit time*: $p_n$, the recorded pit time;
# - *pre-pit flying lap*: $l_{n-1}$, the laptime on the lap preceding the inlap;
# - *post-pit flying lap*: $l_{n+2}$, the laptome on the lap following the outlap;
# - *inlap loss*: $l_n - l_{n-1}$, the time delta between the inlap and the flying lap preceding it;
# - *outlap pre-loss*: $l_{n+1}- l_{n-1}$, the delta between the outlap and the flying lap preceding the inlap;
# - *outlap post-loss*: $l_{n+1}- l_{n+2}$, the delta between the outlap and the flying lap following the outlap;
# - *flying_effect*: $l_{n+2} - l_{n-1}$, the difference between flying laps after the outlap and before the inlap;
# - *additional loss*: $(lap_n + lap_{n+1}) - p_n - (lap_{n-1}+lap_{n+2})$, the additional laptime loss associated with the inlap and outlap having taking account of the recorded pit-time, relative to the flying laps before the inlap and after the outlap.
#
# It would also be useful to record whether or not a pit involved a driver change and which driver was associated with the inlap and outlap.
#
# As well as looking at simple laptimes, we could go further and start to analyse pit events using the finer grained sector times ($s_{i,n}$, for sector $i$ on lap $n$) and driver information. Note however that information about what happened during a pit stop (tyre change, refuelling, repairs, etc) is not available.
#
# Team analysis might explore the role of each particular driver in trying to develop a pit stop time model, not only in terms of their individual inlap and outlap performances, but also in comparison to other drivers for pit events where there is a driver change. (This might also include looking at pit stop times for each ordering of driver change, for example, driver 1 replaces 2, vs driver 2 replaces driver 1.)
#



# # Pit stops - NN - supervised
#
#
# maybe an opportunity to start doing some simple supervised deep learning model predictions. For example, can we detect pitsop from eg five consecutive laptimes?
#
# Then extend to a 2d grid of times (car vs lap) to see if we can detect safety car (where do we get training signal?)


