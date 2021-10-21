#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:25:48 2021

@author: macbookair
"""
class RegObjet :
    
    # Pour chaque algorithme, recenser les parametres qui permettent de mesurer les
    # performance du modéle, et les considérer comme étant les attribut de l'objet.
    
    def __init__(self, coefs, intercept):
        self.coefs = [] # array ([[]])
        self.intercept = None
        
    def affichage(self):
        print("coefs: ", self.coefs,"\n intercept: ", self.intercept)