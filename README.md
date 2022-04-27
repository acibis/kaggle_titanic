# Kaggle Titanic - jak to zrobić, żeby przerobić

*CO:  szybkie i podstawowe przejście przez kagglowy zbiór danych [Titanic](https://www.kaggle.com/c/titanic) - podobno najpierwszy i najlepszy start dla wszystkich chcących ogarnąć Machine Learning. Jeśli szukasz czegoś bardziej skomplikowanego, zajrzyj do [części rozszerzonej](https://github.com/acibis/kaggle_titanic_advanced)*

*PO CO:  żebym w końcu miała w jednym miejscu skrót procesu myślowego, który zaszedł w trakcie pracy nad zbiorem. Plus to taki mój mały pamiętniczek, zmusza mnie do myślenia, sprawdzania pojęć, definicji. Więc po co - totalnie dla mnie ;)*

*DLACZEGO PO POLSKU:  bo tutoriali w języku angielskim na temat Titanica jest już milion pięćset sto dziewięćset, a polskich prawie wcale*

## 1. WSTĘP

<p align="justify"> Żeby zacząć pracować z danymi i Machine Learningiem, należy posiadać bardzo specyficzny rys psychologiczny. Trzeba się pogodzić ze stałą frustracją, poszukiwaniem, wiecznym cofaniem się o kilka kroków, niejednokrotnie niemożnością ułożenia wszystkiego chronologicznie, logicznie i ładnie. Pamiętam, jak na samym początku drogi denerwowałam się, widząc długaśne analizy danych, wykresy i ściany tekstu, podczas gdy szukałam przecież MLa, a nie statystyki. I pamiętam irytację, kiedy magiczny ML okazywał się trzema nieopisanymi linijkami kodu w gąszczu czyszczenia danych, feature engineeringu i innych rzeczy. Teraz, długi czas później, już wiem, że to jednak te wszystkie poprzedzające etapy są najważniejsze, a ML to rzeczywiście trzy nieimponujące linijki, które wykonują 10% pracy, a których powodzenie zależy od 90% reszty etapów. Również dużo czasu zajęło mi zrozumienie, że warto cenić prostotę i zaczynać od najmniej skomplikowanych pomysłów. A zatem: do najprostszego możliwego rozwiązania tego problemu!

Co jest zawsze ważne?
 - model ML przyjmuje jedynie dane liczbowe - jeśli dane mają inną formę (na przykład kolor), należy je przerobić
 - model ML nie przyjmie danych z brakami - jeśli dane mają braki, należy się ich pozbyć (usunąć lub uzupełnić, wedle uznania)
 - jedne cechy mają większy wpływ na przewidywany wynik niż inne - niektórych wcale nie trzeba brać pod uwagę i wrzucać do modelu
 - można tworzyć nowe cechy na podstawie już istniejących. Jak?
 - przed przystąpieniem do tworzenia modelu dobrze jest przeanalizować dane i użyć zdrowego rozsądku oraz intelektu (niestety)
 - warto połączyć zbiór treningowy z testowym, żeby zwiększyć ilość danych.
  
  ## 2. PLAN
  1. Wczytać dane, połączyć zbiory i rzucić na nie okiem.
  2. Ogarnąć kontekst historyczno-kulturowy.
  3. Przygotować/wyczyścić dane.
  4. Stworzyć model i sprawdzić jego skuteczność.
  5. Wgrać wynik do Kaggle.
 
 
  ### 2.1 Wczytać dane, połączyć zbiory i rzucić na nie okiem.
  
```
#import danych treningowych
df_train = pd.read_csv("./train.csv")
#import danych testowych
df_test = pd.read_csv("./test.csv")

# połączenie zbiorów
df = df_train.append(df_test, ignore_index=True, sort=True)
df.head(5)
 ```
![image](https://user-images.githubusercontent.com/13216011/148648536-4fc0ac60-2971-4f25-94ef-4db089740aef.png)

  
  Legenda:
  ![image](https://user-images.githubusercontent.com/13216011/148647782-be0cb08f-19c3-4e7f-82f2-50b07a722d45.png)


 
  ### 2.2 Kontekst historyczny.

Dobrze jest mieć szerszy wgląd w sytuację i dane, które analizujemy. Tutaj ważniejsze fragmenty artykułu z Wikipedii:

>"RMS Titanic – brytyjski transatlantyk typu Olympic, który w nocy z 14 na 15 kwietnia 1912 roku, podczas dziewiczego rejsu na trasie Southampton – Cherbourg – Queenstown – Nowy Jork, zderzył się z górą lodową i zatonął.

>Dane o ofiarach są niejednoznaczne – w zależności od źródeł. Spośród 2208–2228 pasażerów i załogi „Titanica” zginęło ponad 1500 osób. Przeżyło katastrofę tylko około 730. Z pasażerów I klasy zginęło nieco mniej niż połowa, z pasażerów II klasy około 60%, z pasażerów III klasy trzy czwarte. Załogi zginęło prawie 80%.

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
    - jeśli posiadamy kolumnę pełną 0 i 1, w której brakuje wartości, możemy wpisać w brakujące miejsca same 0, lub same 1. Jest to o tyle zdroworozsądkowe, że w kolumnie występują 0 i 1 (nie wpisujemy 3jek), ale o tyle małorozsądkowe, że nasza bonusowa liczba 0 lub 1 może zmienić wynik końcowy. Możemy więc wejść na wyższy poziom zdrowego rozsądku, sprawdzić stosunek 0 do 1 i wpisać odpowiednio w brakujące miejsca od góry najpierw pewną liczbę 0 , a potem pewną liczbę 1 odpowiadającą stosunkowi w całej kolumnie. Czyli jeśli mamy 90 wierszy, w których jest 60 zer i 20 jedynek, oraz 10 brakującyh pól, to wpisujemy tam 7 zer i 3 jedynki.
    - jeśli mamy kolumnę zawierającą czyjś wiek ( a to się za chwilę wydarzy), to w brakujące miejsca możemy: wpisać dowolną wartość z zakresu długości ludzkiego życia (zdrowy rozsądek podpowiada, żeby to nie było -4 albo 236). Na przykład 2 (bo dlaczego nie 2). Możemy też sprawdzić, jaki jest średni wiek wszystkich ludzi w zbiorze. Albo mediana wieku. Albo możemy sprawdzić, jaki jest rozkład odpowiednich grup wiekowych i na tej podstawie uzupełnić pola tak, żeby stosunek poszczególnych grup (dzieci, nastolatkowie, dorośli, emeryci) został zachowany.
  
  Zatem do dzieła. Gdzie brakuje nam danych?
  ```
  # brakujące wartości
  df.isnull().sum()
  ```
  
![image](https://user-images.githubusercontent.com/13216011/165539802-a0fd6b41-cafd-455d-a49a-3453b93a6bc9.png)

  Nie jest źle. Zacznijmy od najmniejszego problemu, czyli brakującej dla 1 osoby informacji o cenie biletu. Uzupełnijmy ten drobny brak średnią ceną wszystkich biletów:
 
 ```
 df.loc[df['Fare'].isnull(), 'Fare'] = 14.435422
 ```
 
 Rozwiązaliśmy problem BRAKUJĄCEJ wartości. Ale kiedy przyjrzymy się kolumnie Fare, dostrzeżemy, że zawiera ona również zera.
 
 ![image](https://user-images.githubusercontent.com/13216011/165541046-5b60ca83-f9a4-4c35-809a-990ed244a123.png)

 Czy to możliwe, żeby ktoś nie zapłacił za bilet? Dostał go w prezencie? Nie, encyklopedia Titanica o niczym takim nie wspomina. Wyliczymy więc średnią cenę biletu dla każdej klasy i portu zaokrętowania i w ten sposób wypełnimy zera. Oczywiście można by to również zastosować do brakującej wartości powyżej - obydwie metody są tak samo skuteczne.
 
 ```
fare_means = df.groupby(['Embarked', 'Pclass'])['Fare'].mean().reset_index()
def find_fare(embarked, pclass):
    fare = fare_means[(fare_means['Embarked'] == embarked) & (fare_means['Pclass'] == pclass)]['Fare'].values[0]
    return(fare)
 
 df['Fare'] = df.apply(lambda x: find_fare(x.Embarked, x.Pclass) if x.Fare == 0 else x.Fare, axis=1)
 ```
 

 Następnie mamy dla 2 osób brak informacji o porcie zaokrętowania. Podglądnijmy te 2 wiersze:
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
 
 Braków w kolumnie Wiek i Kabina mamy dużo, a nawet o wiele więcej. W tym podstawowym rozwiązaniu podejmiemy decyzję, żeby te kolumny w takim razie porzucić. 
 
 ![image](https://user-images.githubusercontent.com/13216011/165540553-1040f060-85fc-4117-9766-03c39a82f401.png)

  
 #### 2.3.2 Zmienne nienumeryczne.
  
  Model ML przyjmuje jedynie wartości numeryczne. Musimy więc wszystkie kolumny, które chcemy do modelu wcisnąć, a które zawierają coś innego niż cyfry, zamienić na formę zrozumiałą dla komputera. W tej chwili nasze dane mają formę:
  ```
  df.info()
  ```
  ![image](https://user-images.githubusercontent.com/13216011/148649627-daeeeebf-86b3-41b7-acb3-2c44377181e6.png)
  
  Imię i nazwisko pasażera raczej nam się nie przyda, tak samo jak numer biletu, te kolumny pominiemy. Zdecydowaliśmy sie też pominąć wiek i kabinę. Potrzebna jest natomiast płeć, port i klasa. Utworzymy więc dodatkowe kolumny dla tych wszystkich cech, zawierające jedynki i zera, czyli wartości liczbowe, których oczekuje komputer, zamiast podawać mu wartości opisowe typu S, C, Q, male, female itd
  
```
type_dummies = pd.get_dummies(df['Sex'])
df = pd.concat([df,type_dummies],axis=1)
 
type_dummies = pd.get_dummies(df['Embarked'])
df = pd.concat([df,type_dummies],axis=1)
 
type_dummies = pd.get_dummies(df['Pclass'])
df = pd.concat([df,type_dummies],axis=1)
 
df.sample()
```
 
 ![image](https://user-images.githubusercontent.com/13216011/165543117-f78c28d8-6464-40d6-8f45-b4671030dff6.png)

Dlaczego dodajemy kolumny również dla klasy, skoro klasa ma wartość liczbową? Owszem, ma, ale ma wartość 1, 2 , 3. # dla komputera jest większe, niż 2 i 1, mógłby więc w jakiś sposób faworyzować trójki. Albo jedynki. Albo wpaść na inny pomysł, widząc dane, które można porównać. Dlatego zamienimy 1, 2 i 3 na dodatkowe kolumny z zerami i jedynkami, tak samo jak płeć i port. To ma sens?

 
 ### 2.4 Model ML.
 ![image](https://user-images.githubusercontent.com/13216011/148962715-1c607d33-f572-4199-847b-f5f9fa5e728a.png)

 MLowa część kodu zwykle jest najkrótsza, dużo dzieje się samo i automagicznie, ale wciąż pozostaje kilka decyzji, które musimy podjąć sami. Między innymi:
 - które zmienne wejdą do modelu (to mamy trochę wyżej, zdecydowaliśmy się użyć zmiennych Klasa, Wiek, Cena, Port, SibSp i Parch. Wiek i Kabina zostały odrzucone ze względu na duże braki, Bilet ze względu na bardzo dużą niejednorodność danych (bilet to kombinacja literek i cyferek, kilkaset różnych rekordów, o których nie wiemy prawie nic), imię, ponieważ jest unikalne dla każdego pasażera, tak samo jak ID, a zatem mało nam mówi o grupach ludzi (grupa przeżył lub nie)
 - jakiego modelu będziemy używać. Nasze badanie oczekuje klasyfikacji, przypisania osoby do grupy 'przeżył' lub 'nie przeżył'. Z tego względu mamy do wyboru szereg modeli do klasyfikacji przenzaczonych, takich jak drzewa losowe, catboost xgboost i inne.
 
 Let's go!
 
 Po pierwsze, metoda odpowiedzialna za trenowanie modelu ma nazwę: **`fit`** i oczekuje  2 argumentów:
- Pierwszy argument to  **macierz/tablica** cech (**Uwaga**: cecha może być jedna, ale to nadal ma być tablica, nie wektor!)
- Drugi argument to **wektor** zmiennej docelowej (eng. *target variable*)
 
 
```
# lista cech, których chcemy użyć w modelu:
feats = ['Fare', 'Parch', 'SibSp', 'female',  'male', 'C',    'Q',   'S',   1,  2, 3]

# macierz cech. Pamiętajmy, że złączyliśmy 2 zbiory! Model chcemy stworzyć na podstawie zbioru treningowego, więc 
# musimy odfiltrować te wiersze, które nie mają zmiennej Survived!

train = df[df['Survived'].isna() == False][feats] # macierz cech zbioru treningowego
targets = df[df['Survived'].isna() == False]['Survived'] # wektor zmiennej docelowej

X_test = df[feats] # macierz cech zbioru testowego

model = xgb.XGBClassifier() # wybieramy model, dziś to będzie XGBoost

model.fit(train, targets) # trenujemy model na zbiorze treningowym
 
predictions = model.predict(X_test) # używamy modelu na zbiorze treningowym i próbujemy 'przewidzieć' zawartość kolumny Survived

df["y_pred"] = predictions # dopisujemy przewidziane wyniki do naszego datasetu
```
 
I to już :)  Wiem, mało imponujące. Co tu zaszło? Algorytm dostał macierz cech X (train) i wektor cechy docelowej Y (targets). Zbadał związki między jednym i drugim i zbudował funkcję, według której cechy train wpływają na wartość targets. Następnie tę funkcję aplikujemy do testowego zbioru (X_test) i generujemy zestaw wyników. 

Czas wziąć wyniki, wrzucić na Kaggle i dowiedzieć się, jaki mamy wynik.

```
results = df[df["Survived"].isna() == True][["PassengerId", "y_pred"]]
results.columns = ["PassengerId", "Survived"]
results["Survived"] = results["Survived"].astype(int)
results.to_csv("very_basic.csv",index=False)
```
![image](https://user-images.githubusercontent.com/13216011/165547121-1efc2953-904c-4265-8a48-c3c341716dff.png)

 
## 3. PODSUMOWANIE.
 
<p align="justify"> Mój wynik to 0.76555. Myślę, że to się plasuje gdzieś w kategorii brązowego medalu, czyli nie było najgorzej, zdecydowanie może być lepiej. Ale hej, tu chodziło o zrozumienie schematu i kolejności wykonywania działań przy zwyczajnym zadaniu MLowym. Na inżynierię cech i inne cuda przyjdzie jeszcze czas.
 
PS> Btw, nie dajcie się zdeprymować wynikami równymi 1 na leaderboardzie Kaggle. To ludzie, którzy oszukiwali - [tutaj](https://www.kaggle.com/carlmcbrideellis/titanic-leaderboard-a-score-0-8-is-great) info na ten temat.
