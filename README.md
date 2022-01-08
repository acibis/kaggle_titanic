# Kaggle Titanic - od początku do końca

*CO:  przejście przez kagglowy zbiór danych [Titanic](https://www.kaggle.com/c/titanic) - podobno najpierwszy i najlepszy start dla wszystkich chcących ogarnąć Machine Learning*

*PO CO:  żebym w końcu miała w jednym miejscu skrót procesu myślowego, który zaszedł w trakcie pracy nad zbiorem*

*DLACZEGO PO POLSKU:  bo tutoriali w języku angielskim na temat Titanica jest już milion pięćset sto dziewięćset, a polskich prawie wcale*

## 1. WSTĘP

<p align="justify"> Żeby zacząć pracować z danymi i Machine Learningiem, należy posiadać bardzo specyficzny rys psychologiczny. Trzeba się pogodzić ze stałą frustracją, poszukiwaniem, wiecznym cofaniem się o kilka kroków, niejednokrotnie niemożnością ułożenia wszystkiego chronologicznie, logicznie i ładnie. Pamiętam, jak na samym początku drogi denerwowałam się, widząc długaśne analizy danych, wykresy i ściany tekstu, podczas gdy szukałam przecież MLa, a nie statystyki. I pamiętam irytację, kiedy magiczny ML okazywał się trzema nieopisanymi linijkami kodu w gąszczu czyszczenia danych, feature engineeringu i innych rzeczy. Teraz, długi czas później, już wiem, że to jednak te wszystkie poprzedzające etapy są najważniejsze, a ML to rzeczywiście trzy nieimponujące linijki, które wykonują 10% pracy, a których powodzenie zależy od 90% reszty etapów.

Co jest ważne?
 - model ML przyjmuje jedynie dane liczbowe - jeśli dane mają inną formę (na przykład kolor), należy je przerobić
 - model ML nie przyjmie danych z brakami - jeśli dane mają braki, należy się ich pozbyć (usunąć lub uzupełnić, wedle uznania)
 - jedne cechy mają większy wpływ na przewidywany wynik niż inne - niektórych wcale nie trzeba brać pod uwagę i wrzucać do modelu
 - można tworzyć nowe cechy na podstawie już istniejących. Jak?
 - przed przystąpieniem do tworzenia modelu dobrze jest przeanalizować dane i użyć zdrowego rozsądku oraz intelektu (niestety)
  
  ## 2. PLAN
  1. Wczytać dane i rzucić na nie okiem.
  2. Ogarnąć kontekst historyczno-kulturowy.
  3. Przygotować/wyczyścić dane.
  4. Przeanalizować dane.
  5. Przygotować dane do wprowadzenia do modelu.
  6. Stworzyć model i sprawdzić jego skuteczność.
  7. Próbować podnieść skuteczność tworząc nowe cechy.
  
  ### 2.1 Wczytać dane i rzucić na nie okiem.
  
```
# import danych
df = pd.read_csv("./train.csv")
# przykładowe 3 wiersze
data.sample(3)
 ```
![image](https://user-images.githubusercontent.com/13216011/148648536-4fc0ac60-2971-4f25-94ef-4db089740aef.png)

  
  Legenda:
  ![image](https://user-images.githubusercontent.com/13216011/148647782-be0cb08f-19c3-4e7f-82f2-50b07a722d45.png)


 
  ### 2.2 Kontekst historyczny.

Dobrze jest mieć szerszy wgląd w sytuację i dane, które analizujemy. Tutaj ważniejsze fragmenty artykułu z Wikipedii:

>"RMS Titanic – brytyjski transatlantyk typu Olympic, który w nocy z 14 na 15 kwietnia 1912 roku, podczas dziewiczego rejsu na trasie Southampton – Cherbourg – Queenstown – Nowy Jork, zderzył się z górą lodową i zatonął.

>Dane o ofiarach są niejednoznaczne – w zależności od źródeł. Spośród 2208–2228 pasażerów i załogi „Titanica” zginęło ponad 1500 osób. Przeżyło katastrofę tylko około 730. Z pasażerów I klasy zginęło nieco mniej niż połowa, z pasażerów II klasy około 60%, z pasażerów III klasy trzy czwarte. Załogi zginęło praiwe 80%.

>W łodziach „Titanica” było miejsce dla ponad 1100 osób, ale wiele z nich było częściowo pustych. Zwłaszcza w pierwszej fazie ewakuacji łodzie odpływały z niewielką liczbą osób. Dopiero w dalszej fazie wypadku łodzie odpływały pełne. Nie podjęto niemal żadnej próby ratowania osób, które znalazły się w wodzie. Jedynie piąty oficer, Harold Lowe, rozdzielił pasażerów ze swej łodzi między inne łodzie i popłynął wydobywać z wody tych, którzy pływali w morzu, ale zrobił to zbyt późno i ocalił tylko kilka osób."

![image](https://user-images.githubusercontent.com/13216011/148606413-b45c3919-2dc8-4183-858f-84bbbe4b1f45.png)

 Po co dorzucam suche fakty z Wiki? Żebyśmy mogli mieć punkt odniesienia. Porównajmy dane z informacjami. Nasz zbiór zawiera 891 wierszy, czyli 40% pasażerów podróżujących Titanikiem. W naszym zbiorze członków brak załogi (informacja potwierdzona, po prostu zbiór ich nie zawiera).
  
  
  ### 2.3 Porządkowanie danych.
  
  #### 2.3.1 Brakujące wartości.
  
  Jednym z problemów, z jakimi mierzymy się w świecie data science, są brakujące dane. Sposobów na radzenie sobie z nim jest kilka i trzeba nimi żonglować w zależności od sytuacji. Jaki więc mamy wybór?
  - usunąć wiersze z brakującymi wartościami. Jeśli wierszy mamy dużo, te, w których występują brakujące wartości nie są wyjątkowe i niewiele stracimy, możemy je usunąć
  - usunąć kolumny z brakującymi wartościami. Jeśli wiemy, że dana kolumna nie będzie brała udziału w analizie (bo nie ma związku i wpływu na przewidywaną przez nas cechę), możemy ją porzucić
  - uzupełnić kolumny byle jaką wartością liczbową. Cokolwiek (na przykład 0) bardziej się spodoba modelowi niż brak wartości. Czasami to nie ma wpływu na wynik, czasami jeśli nam się spieszy i chcemy po prostu mieć działający (nie mówię, że dobrze działający) model, można skorzystać z tej metody.
  - uzupełnić wiersze własnymi wartościami używając zdrowego rozsądku. Na przykład:
    - jeśli posiadamy kolumnę pełną 0 i 1, w której brakuje wartości, możemy wpisać w brakujące miejsca same 0, lub same 1. Jest to o tyle zdroworozsądkowe, że w kolumnie występują 0 i 1 (nie wpisujemy 3jek), ale o tyle małorozsądkowe, że nasza bonusowa liczba 0 lub 1 może zmienić wynik końcowy. Możemy więc wejść na wyższy poziom zdrowego rozsądku, sprawdzić stosunek 0 do 1 i wpisać odpowiednio w brakujące miejsca od góry najpierw pewną liczbę 0 , a potem pewną liczbę 1 odpowiadającą stosunkowi w całej kolumnie. Czyli jeśli mamy 100 wierszy, w których jest 60 zer i 20 jedynek, oraz 10 brakującyh pól, to wpisujemy tam 6 zer i 2 jedynki.  Na wyżyny zdrowego rozsądku wznosimy się uzupełniając losowo brakujące miejsca zerami i jedynkami w stosunku odpowiednim do ich liczby w całej kolumnie.
    - jeśli mamy kolumnę zawierającą czyjś wiek ( a to się za chwilę wydarzy ), to w brakujące miejsca możemy: wpisać dowolną wartość z zakresu długości ludzkiego życia (zdrowy rozsądek podpowiada, żeby to nie było -4 albo 236). Na przykład 2 (bo dlaczego nie 2). Możemy też sprawdzić, jaki jest średni wiek wszystkich ludzi w zbiorze. Albo mediana wieku. Albo możemy sprawdzić, jaki jest rozkład odpowiednich grup wiekowych i na tej podstawie uzupełnić polatak, żeby stosunek poszczególnych grup (dzieci, nastolatkowie, dorośli, emeryci) został zachowany.
  
  Zatem do dzieła. Gdzie brakuje nam danych?
  ```
  # brakujące wartości
  df.isnull().sum()
  ```
  
  ![image](https://user-images.githubusercontent.com/13216011/148649032-b081e9d1-d8f2-44d8-9109-75064b0aa64c.png)

  Nie jest źle. Zacznijmy od najmniejszego problemu, czyli brakującej dla 2 osób informacji o porcie zaokrętowania. Podglądnijmy te 2 wiersze:
  ```
  df[df['Embarked'].isnull()]
  ```
  ![image](https://user-images.githubusercontent.com/13216011/148649063-4cee0c94-59fd-4608-84a7-866937f32f1a.png)
Widzimy, że te dwie podróżne mieszkały w jednej kabinie. Być może płynęły z kimś jeszcze, kto mieszkał w tej samej kabinie i posiada informację o porcie, w którym wsiedli?
 ```
 df[df['Cabin'] == "B28"]
 ```
  ![image](https://user-images.githubusercontent.com/13216011/148649125-8539ade6-eedb-40e2-8529-63617cdf95f5.png)
Niestety nie. Skorzystajmy więc po prostu z potęgi internetu, wpiszmy w wyszukiwarkę imię i nazwisko pasażerki:
  
  >Miss Rose Amélie Icard, 38, was born in Vaucluse, France on 31 October 1872, her father Marc Icard lived at Mafs á Murs (?).
She boarded the Titanic at Southampton as maid to Mrs George Nelson Stone. She travelled on Mrs Stone's ticket (#113572).
  
  Problem rozwiązany, dorzucamy 'S' w brakujące miejsca:
  ```
  df.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'
  ```
  
  Następnie kwestia wieku - brakuje nam 177 z 891 wartości i tak, jak pisałam wyżej, proponuję chwilowo uzupełnić braki najczęsciej wystepującą w kolumnie wiek wartością.
  ```
  df.loc[df['Age'].isnull(), 'Age'] = df['Age'].median()
  ```
  
  Ostatnie brakujące wartości dotyczą numeru kabiny. Brakuje większości, bo aż 687 z 891 numerów. W tej sytuacji, chwilowo decyduję się tę kolumnę pominąć w analizie i modelu.
  
    #### 2.3.2 Zmienne nienumeryczne.
  
  Model ML przyjmuje jedynie wartości numeryczne. Musimy więc wszystkie kolumny, które chcemy do modelu wcisnąć, a które zawierają coś innego niż cyfry, zamienić na formę zrozumiałą dla komputera. W tej chwili nasze dane mają formę:
  ```
  df.info()
  ```
  ![image](https://user-images.githubusercontent.com/13216011/148649627-daeeeebf-86b3-41b7-acb3-2c44377181e6.png)
  
  Imię i nazwisko pasażera raczej nam się nie przyda, tak samo jak numer biletu, te koumny pominiemy. Potrzena jest natomiast płeć i port. Zamieńmy płeć (male, female) na 0 i 1, a port (S, C, Q) na 1, 2 i 3.
  
 ```
 dict = {"S" : 1, "C" : 2, "Q": 3}

df['Embarked_Int'] = df['Embarked']
df = df.replace({"Embarked_Int": dict})
df['Sex_Int'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)
```

  ### 2.4 Analiza danych.

  Zbudujmy kilka wykresów, żeby lepiej widziec zależności.
  
  ```
import matplotlib.pyplot as plt   
%matplotlib inline

import seaborn as sns

fig, ax = plt.subplots(2,3, figsize=(20,8))
sns.barplot(x="Sex", y="Survived", palette='flare', data=df, ax=ax[0][0]).set_title('Płeć vs Przetrwanie')
sns.barplot(x="Pclass", y="Survived", palette='flare', data=df, ax=ax[0][1]).set_title('Klasa vs Przetrwanie')
sns.barplot(x="Parch", y="Survived", palette='flare', data=df, ax=ax[0][2]).set_title('Dzieci/Rodzice vs Przetrwanie')
sns.barplot(x="SibSp", y="Survived", palette='flare', data=df, ax=ax[1][0]).set_title('Krewni vs Przetrwanie')
sns.barplot(x="Embarked", y="Survived", palette='flare', data=df, ax=ax[1][1]).set_title('Port vs Przetrwanie')
sns.histplot(x="Age", y='Survived', palette='flare', data=df, ax=ax[1][2]).set_title('Wiek vs Przetrwanie')
```  
  ![image](https://user-images.githubusercontent.com/13216011/148649819-76de1297-e334-4e68-8285-6e5e5bccc4ce.png)

 Z powyższych wykresów wynika, że:

kobiety mają większą szansę na przetrwanie
pasażerowie pierwszej klasy mają wiekszą szansę na przetrwanie
podróżujący z dziećmi/rodzicami mają większą szansę na przetrwanie (chyba, że masz powyżej 3jki dzieci)
podróżujący ze współmałżonkiem lub rodzeństwem mają większe szanse na przetrwanie
osoby w wieku 15-30 mają mniejsze szanse na przetrwanie
