# This will take a while to run
def keyword_combs(r, thres = 0):
    ''' Inputs combinations of length r and visualizes only the
    combinations that occur higher than threshold thres '''
    # Storing the keyword comb and counts here
    keyword_overlaps = {}
    
    # Iterating through every combination and elements in that tuple
    comb_lengths = []
    comb_tweet_sets = []
    for tup in list(itertools.combinations(intents.keys(), r)):
        for i in range(r):
            comb_tweet_sets.append(to_set(get_key_tweets(processed_inbound, intents[tup[i]])))
    #     print(intents[tup[0]], intents[tup[1]])
        
    
    # Filtering to just the significant ones, which I define as greater than 100
    combs = []
    counts = []
    for i in keyword_overlaps.items():
        if i[1] > 100:
            combs.append(i[0])
            counts.append(i[1])

    # Visualizing as well
    v = pd.DataFrame({'Combination': combs, "Counts": counts}).sort_values('Counts', ascending = False)
    plt.figure(figsize=(9,6))
    sns.barplot(x = v['Combination'], y = v['Counts'], palette = 'magma')
    plt.title(f'Combinations of 2 Keywords (At Least {thres} Occurances)')
    plt.xticks(rotation=90)
    plt.show()
    # Getting the lengths of the intersections of the tweet sets in a certain intent
    print(comb_tweet_sets)

