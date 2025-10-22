import sys
import random
import math
import os
from operator import itemgetter
import easygui
from collections import defaultdict
import numpy as np
import pandas as pd

import copy
from math import pow  # 主要用于求平方

import matplotlib.pyplot as plt


random.seed(0)

class UserBasedCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self):
        self.data = []

        self.trainset = {}


        self.game_name = {}

        self.n_sim_user = 20
        self.n_rec_game = 10

        self.user_sim_mat = {}
        self.game_popular = {}
        self.game_count = 0

        print ('Similar user number = %d' % self.n_sim_user, file=sys.stderr)
        print ('recommended game number = %d' %
               self.n_rec_game, file=sys.stderr)


    def generate_dataset(self,filename):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        s = pd.read_csv(filename, header=None)
        self.data = np.array(s)


        for user, game, action, value, _ in self.data:
            # split the data by pivot
            if action == 'purchase':

                self.trainset.setdefault(user, {})
                self.trainset[user][game] = 0
                trainset_len += 1
            else:
                self.trainset[user][game] += value


        print ('split training set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)

    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=movieID, value=list of userIDs who have seen this movie
        print ('building movie-users inverse table...', file=sys.stderr)
        game2users = dict()

        for user, games in self.trainset.items():
            for game in games:
                # inverse table for item-users
                if game not in game2users:
                    game2users[game] = set()
                game2users[game].add(user)
                # count item popularity at the same time
                if game not in self.game_popular:
                    self.game_popular[game] = 0
                self.game_popular[game] += 1
        print ('build game-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.game_count = len(game2users)
        print ('total game number = %d' % self.game_count, file=sys.stderr)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print ('building user co-rated movies matrix...', file=sys.stderr)

        for game, users in game2users.items():
            for u in users:
                usersim_mat.setdefault(u, defaultdict(int))
                for v in users:
                    if u == v:
                        continue
                    usersim_mat[u][v] += 1
        print ('build user co-rated game matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar users and recommend N movies. '''
        K = self.n_sim_user
        N = self.n_rec_game
        rank = dict()
        watched_games = self.trainset[user]

        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for game in self.trainset[similar_user]:
                if game in watched_games:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(game, 0)
                rank[game] += similarity_factor
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_game

        print('ok')
        a = 1
        while int(a) > 0:
            # a = input('please input the number of users to recommend: ')
            a = easygui.enterbox("please input the number of users to recommend: ")
            if a == None:
                break
            if int(a) <= 0:
                break

            user_test = self.data[int(a)][0]
            rec_games_2 = self.recommend(user_test)
            str = ''
            for game, _ in rec_games_2:
                # print(int(a))
                # print(self.movie_name[movie])
                str += game
                str += '\n'
            easygui.msgbox(str)




class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}

        self.data = []

        self.game_name = {}

        self.n_sim_game = 20
        self.n_rec_game = 10

        self.game_sim_mat = {}
        self.game_popular = {}
        self.game_count = 0

        print('Similar game number = %d' % self.n_sim_game, file=sys.stderr)
        print('Recommended game number = %d' %
              self.n_rec_game, file=sys.stderr)



    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        s = pd.read_csv(filename, header=None)
        self.data = np.array(s)

        for user, game, action, value, _ in self.data:
            # split the data by pivot
            if action == 'purchase':

                self.trainset.setdefault(user, {})
                self.trainset[user][game] = 0
                trainset_len += 1
            else:
                self.trainset[user][game] += value

        print ('split training set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)

    def calc_movie_sim(self):
        ''' calculate game similarity matrix '''
        print('counting game number and popularity...', file=sys.stderr)

        for user, games in self.trainset.items():
            for game in games:
                # count item popularity
                if game not in self.game_popular:
                    self.game_popular[game] = 0
                self.game_popular[game] += 1

        print('count game number and popularity succ', file=sys.stderr)

        # save the total number of movies
        self.game_count = len(self.game_popular)
        print('total game number = %d' % self.game_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.game_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        for user, games in self.trainset.items():
            for m1 in games:
                itemsim_mat.setdefault(m1, defaultdict(int))
                for m2 in games:
                    if m1 == m2:
                        continue
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)

        # calculate similarity matrix
        print('calculating game similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_games in itemsim_mat.items():
            for m2, count in related_games.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.game_popular[m1] * self.game_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating game similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)

        print('calculate game similarity matrix(similarity factor) succ',
              file=sys.stderr)
        print('Total similarity factor number = %d' %
              simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar games and recommend N games. '''
        K = self.n_sim_game
        N = self.n_rec_game
        rank = {}
        watched_games = self.trainset[user]

        for game, rating in watched_games.items():
            for related_game, similarity_factor in sorted(self.game_sim_mat[game].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:
                if related_game in watched_games:
                    continue
                rank.setdefault(related_game, 0)
                rank[related_game] += similarity_factor * rating
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_game


        print('ok')
        a = 1
        while int(a) > 0:
            # a = input('please input the number of users to recommend: ')
            a = easygui.enterbox("please input the number of users to recommend: ")
            if a == None:
                break
            if int(a) <= 0:
                break
            user_test = self.data[int(a)][0]
            rec_games_1 = self.recommend(user_test)
            str = ''
            for game, _ in rec_games_1:
                # print(int(a))
                # print(self.movie_name[movie])
                str += game
                str += '\n'
            easygui.msgbox(str)



if __name__ == '__main__':

    msg = "which method would you like to try"
    title = "choose a method"
    choices = ["User-based collaborative filtering", "Item-based collaborative filtering"]

    choice = easygui.choicebox(msg, title, choices)

    if choice=="User-based collaborative filtering":
        usercf = UserBasedCF()
        usercf.generate_dataset('steam-200k.csv')
        usercf.calc_user_sim()
        usercf.evaluate()

    if choice=="Item-based collaborative filtering":
        itemcf = ItemBasedCF()
        itemcf.generate_dataset('steam-200k.csv')
        itemcf.calc_movie_sim()
        itemcf.evaluate()
