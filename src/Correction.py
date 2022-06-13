from numpy.random import rand, randint
import re
import os

def estdans(t, v):  # Recherche par dichotomie dans un tableau trié
    a = 0
    b = len(t)
    if b == 0:
        return False
    while b > a + 1:
        m = (a + b) // 2
        if t[m] > v:
            b = m
        else:
            a = m
    return t[a] == v


def estdans_index(t, v):  # Recherche par dichotomie dans un tableau trié
    a = 0
    b = len(t)
    if b == 0:
        return False
    while b > a + 1:
        m = (a + b) // 2
        if t[m] > v:
            b = m
        else:
            a = m
    return (t[a] == v, a)


def texte_correct(t, v):
    tab = v.split(" ")
    for mot in tab:
        if (not estdans(t, mot)):
            return False
    return True


def formater(ligne):  # Formatage vers un format plus simple pour la correction
    ligne = ligne.lower()
    ligne = ligne.replace("à", "a")
    ligne = ligne.replace("é", "e")
    ligne = ligne.replace("è", "e")
    ligne = ligne.replace("ê", "e")
    ligne = ligne.replace("â", "a")
    ligne = ligne.replace("ù", "u")
    ligne = ligne.replace("î", "i")
    ligne = ligne.replace("ü", "u")
    ligne = ligne.replace("ï", "i")
    ligne = ligne.replace("ô", "o")
    ligne = ligne.replace("\n", "")
    ligne = ligne.replace("-", "")
    ligne = ligne.replace("\'", " ")
    ligne = ligne.replace(".", " ")
    ligne = ligne.replace(",", " ")
    ligne = ligne.replace("?", " ")
    ligne = ligne.replace(":", " ")
    ligne = ligne.replace(";", " ")
    ligne = ligne.replace("!", " ")
    ligne = ligne.replace("ç", "c")
    ligne = ligne.replace("œ", "oe")
    ligne = ligne.replace("’", " ")
    ligne = ligne.replace("(", " ")
    ligne = ligne.replace(")", " ")
    return ligne


def corriger(phrase):
    tbmots = phrase.split(" ")
    phrase_corrigée = ""
    i = 0
    while i < len(tbmots):
        mot = tbmots[i]
        if (mot != ""):

            if (estdans(dico, mot)):
                phrase_corrigée = phrase_corrigée + " " + mot
                i += 1
            else:
                e2réussie = False
                if (i + 1 < len(tbmots) and len(mot) + len(tbmots[i + 1]) < seuil):
                    if (tbmots[i + 1] == ""):
                        acorriger = mot
                    else:
                        acorriger = mot + " " + tbmots[i + 1]

                    if (texte_correct(dico, acorriger)):
                        phrase_corrigée = phrase_corrigée + " " + acorriger
                        e2réussie = True
                        i = i + 2


                    else:
                        pdm = proches_double_mot(mot, tbmots[i + 1])
                        # proches = proches_itérés (pdm,1)
                        proches_corrects = extraire_proches_corrects(pdm, dico)
                        if (len(proches_corrects) > 0):
                            co = trouvermeilleurcorrection(proches_corrects)[1]
                            phrase_corrigée = phrase_corrigée + " " + co
                            # print(acorriger + " -> " + co)
                            e2réussie = True
                            i = i + 2
                        else:
                            proches = proches_itérés(pdm, 1)
                            proches_corrects = extraire_proches_corrects(pdm, dico)
                            if (len(proches_corrects) > 0):
                                co = trouvermeilleurcorrection(proches_corrects)[1]
                                phrase_corrigée = phrase_corrigée + " " + co
                                # print(acorriger + " -> " + co)
                                e2réussie = True
                                i = i + 2

                if (not e2réussie):
                    acorriger = mot
                    if (texte_correct(dico, acorriger)):
                        phrase_corrigée = phrase_corrigée + " " + acorriger
                    else:
                        proches = proches_itérés([(1, acorriger)], 1)
                        proches_corrects = extraire_proches_corrects(proches, dico)
                        if (len(proches_corrects) > 0):
                            co = trouvermeilleurcorrection(proches_corrects)[1]
                            phrase_corrigée = phrase_corrigée + " " + co
                            # print(acorriger + " -> " + co)
                        else:
                            proches = proches_itérés(proches, 1)
                            proches_corrects = extraire_proches_corrects(proches, dico)
                            if (len(proches_corrects) > 0):
                                co = trouvermeilleurcorrection(proches_corrects)[1]
                                phrase_corrigée = phrase_corrigée + " " + co
                                # print(acorriger + " -> " + co)

                            else:
                                phrase_corrigée = phrase_corrigée + " " + mot
                    i += 1
        else:
            i += 1

            """if(i<(len(tbmots)-1) and (!estdans(dico,tbmots[i+1])) and (estdans(dico,tbmots[i]+tbmots[i+1])) ):
                        phrase_corrigée = phrase_corrigée + " " + tbmots[i]+tbmots[i+1]
            phrase_corrigée = phrase_corrigée+" " +(mot)"""
    return phrase_corrigée


def trouvercorrections(mot):
    corrections = []
    original = mot
    for i in range(0, len(mot)):
        for lettre in alphabet:
            l = list(original)
            l[i] = lettre
            motproche = ("".join(l))
            if (texte_correct(dico, motproche)):
                corrections.append((poidsubstitution(lettre), motproche))
    return corrections


def proches(texte, poid_de_base):
    tableau_proches = []
    for i in range(0, len(texte) + 1):
        for lettre in alphabet:
            if (i < len(texte)):
                l = list(texte)
                l[i] = lettre
                tableau_proches.append((poid_de_base * poidsubstitution(lettre), "".join(l)))
            l = list(texte)
            tableau_proches.append((poid_de_base * poidajout(lettre), "".join(l[0:i] + [lettre] + l[i:len(texte)])))

    return tableau_proches


def proches_itérés(tbmots, n):
    if (n == 0):
        return tbmots
    else:
        tb = tbmots
        for (poid, mot) in tbmots:
            tb = tb + proches(mot, poid)
        return proches_itérés(tb, n - 1)


def proches_double_mot(mot1, mot2):
    tbproches = []
    for i in range(0, 26):
        tbproches.append((poidsubstitution(alphabet[i]), mot1 + alphabet[i] + mot2))
    return tbproches


def extraire_proches_corrects(proches, dico):
    tb = []
    for (poid, mot) in proches:
        if (texte_correct(dico, mot)):
            ms = mot.split(" ")
            f = 1
            for m in ms:
                (b, i) = estdans_index(dico, m)
                f = f * (1 + mot_frequences[i]) / nombre_mots
            tb.append((poid * f, mot))
    return tb


def trouvermeilleurcorrection(corrections):
    index = 0
    max = corrections[0][0]
    for i in range(1, len(corrections)):
        if (corrections[i][0] > max):
            index = i
            max = corrections[i][0]
    return corrections[index]


def poidajout(lettre):
    f = frequence(lettre)
    # return 1
    return C1 * f / 100


def poidsubstitution(lettre):
    f = frequence(lettre)
    return C2 * f / 100
    # return 1/2


def simulateur(texte, proba):
    texte = formater(texte)
    sortie = ""
    for lettre in list(texte):
        if (rand() <= proba):
            sortie = sortie + alphabet[randint(len(alphabet))]
        else:
            sortie = sortie + lettre
    return sortie


def frequence(lettre):
    index = alphabet.find(lettre)
    return frequences[index]

print(os.getcwd())
dicotxt = open("../dico_frequence.txt", "r", encoding='utf8')
ligne = dicotxt.readline()
dico = []
mot_frequences = []
nombre_mots = 0

while (ligne):
    (mot, freq) = ligne.split("|")

    dico.append(formater(mot))
    mot_frequences.append(int(freq))
    nombre_mots += int(freq)
    ligne = dicotxt.readline()

alphabet = "azertyuiopqsdfghjklmwxcvbn "
frequences = [6.34, 0.13, 12.3, 5.16, 5.04, 0.39, 3.82, 5.62, 4.27, 2.12, 0.55, 5.54, 3.12, 0.94, 1.05, 0.94, 0.29,
              0.25, 4.22, 2.23, 0.14, 0.32, 2.71, 0.94, 0.97, 5.44, 17.54]
C1 = 1  # probabilité d'oublier une lettre
C2 = 1  # probabilité de substituer une lettre
seuil = 10

""" Example use
texte = "Le bon sens est la chose du monde la mieux partagée ; car chacun pense en être si bien pourvu, que ceux même qui sont les plus difficiles à contenter en toute autre chose n’ont point coutume d’en désirer plus qu’ils en ont. En quoi il n’est pas vraisemblable que tous se trompent : mais plutôt cela témoigne que la puissance de bien juger et distinguer le vrai d’avec le faux, qui est proprement ce qu’on nomme le bon sens ou la raison, est naturellement égale en tous les hommes "

texte1 = "a plaine rase sous la quit sans poiles dune obscurit et dune paisseur dencre un homme quivait seul la grande route de marchiennes  montsou dix kilompres de pav coupant pout droit  travers les champs de betteraves devant lui il de voyait mme pas le sol joir et il navait la sensation de limmense horison plat que par les"
acorriger = simulateur(formater(texte), 0.1)

print("ENTREE : ")
print(acorriger)
print("_____________")
print("CORRECTION : ")
print(corriger(acorriger))
"""