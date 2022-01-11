# Kaggle Titanic - jak to zrobić, żeby przerobić

*CO:  szybkie przejście przez kagglowy zbiór danych [Titanic](https://www.kaggle.com/c/titanic) - podobno najpierwszy i najlepszy start dla wszystkich chcących ogarnąć Machine Learning*

*PO CO:  żebym w końcu miała w jednym miejscu skrót procesu myślowego, który zaszedł w trakcie pracy nad zbiorem. Plus to taki mój mały pamiętniczek, zmusza mnie do myślenia, sprawdzania pojęć, definicji. Więc po co - totalnie dla mnie ;)*

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
  5. Stworzyć model i sprawdzić jego skuteczność.
  6. Wgrać wynik do Kaggle.
 
 
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
    - jeśli posiadamy kolumnę pełną 0 i 1, w której brakuje wartości, możemy wpisać w brakujące miejsca same 0, lub same 1. Jest to o tyle zdroworozsądkowe, że w kolumnie występują 0 i 1 (nie wpisujemy 3jek), ale o tyle małorozsądkowe, że nasza bonusowa liczba 0 lub 1 może zmienić wynik końcowy. Możemy więc wejść na wyższy poziom zdrowego rozsądku, sprawdzić stosunek 0 do 1 i wpisać odpowiednio w brakujące miejsca od góry najpierw pewną liczbę 0 , a potem pewną liczbę 1 odpowiadającą stosunkowi w całej kolumnie. Czyli jeśli mamy 90 wierszy, w których jest 60 zer i 20 jedynek, oraz 10 brakującyh pól, to wpisujemy tam 6 zer i 2 jedynki.  Na wyżyny zdrowego rozsądku wznosimy się uzupełniając losowo brakujące miejsca zerami i jedynkami w stosunku odpowiednim do ich liczby w całej kolumnie.
    - jeśli mamy kolumnę zawierającą czyjś wiek ( a to się za chwilę wydarzy), to w brakujące miejsca możemy: wpisać dowolną wartość z zakresu długości ludzkiego życia (zdrowy rozsądek podpowiada, żeby to nie było -4 albo 236). Na przykład 2 (bo dlaczego nie 2). Możemy też sprawdzić, jaki jest średni wiek wszystkich ludzi w zbiorze. Albo mediana wieku. Albo możemy sprawdzić, jaki jest rozkład odpowiednich grup wiekowych i na tej podstawie uzupełnić pola tak, żeby stosunek poszczególnych grup (dzieci, nastolatkowie, dorośli, emeryci) został zachowany.
  
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
  
  Imię i nazwisko pasażera raczej nam się nie przyda, tak samo jak numer biletu, te koumny pominiemy. Potrzebna jest natomiast płeć i port. Zamieńmy płeć (male, female) na 0 i 1, a port (S, C, Q) na 1, 2 i 3.
  
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

- kobiety mają większą szansę na przetrwanie
- pasażerowie pierwszej klasy mają wiekszą szansę na przetrwanie
- podróżujący z dziećmi/rodzicami mają większą szansę na przetrwanie (chyba, że masz powyżej 3jki dzieci)
- podróżujący ze współmałżonkiem lub rodzeństwem mają większe szanse na przetrwanie
 
 Oprócz zrobienia wykresów i naocznego badania zależności, możemy też sprawdzić korelację (tutaj zwyczajowo przypominamy sobie, że korelacja mówi o tym, czy dwie zmienne są ze sobą istotnie statystycznie powiązane, a nie o tym, że jedna powoduje drugą) poszczególnych kolumn z kolumną 'Survived':
 ```
 df.corrwith(df['Survived']).sort_values()
```
 
![image](https://user-images.githubusercontent.com/13216011/148959424-c578c058-34c7-4b83-895d-54d9d6f31b9c.png)

Jak widać wyżej, wysoką korelację (ujemną lub dodatnią) mają płeć, klasa, opłata za bilet i port zaokrętowania - te zmienne wprowadzimy do modelu.(Może nieco dziwić niska korelacja wieku, ale w tym momencie nie będziemy się tym przejmować.)
 
 ### 2.5 Model ML.
 ![image](https://user-images.githubusercontent.com/13216011/148962715-1c607d33-f572-4199-847b-f5f9fa5e728a.png)

 MLowa część kodu zwykle jest najkrótsza, dużo dzieje się samo i automagicznie, ale wciąż pozostaje kilka decyzji, które musimy podjąć sami. Między innymi:
 - które zmienne wejdą do modelu
 - jakiego modelu będziemy używać
 - jak będziemy testować jego skuteczność
 
 Let's go!
 
 Po pierwsze, metoda odpowiedzialna za trenowanie modelu ma nazwę: **`fit`** i oczekuje  2 argumentów:
- Pierwszy argument to  **macierz/tablica** cech (**Uwaga**: cecha może być jedna, ale to nadal ma być tablica, nie wektor!)
- Drugi argument to **wektor** zmiennej docelowej (eng. *target variable*)
 
 Przygotowujemy zmienne (te same zmienne, o których wiemy, że mają istotną korelację ze zmienną docelową Survived):
 ```
feats = [ 'Sex_Int', 'Pclass', 'Fare', 'Embarked' ] #zmienne, które mają wysoką korelację z Survived

X = df[feats].values # X to nasza macierz wartości (uwaga suchar, jak się nazywa męska tablica? pacierz)

y = df["Survived"].values # wektor zmiennej docelowej
 ```
 
 Gdyby ktoś się zastanawiał, X wygląda tak:
 ![image](https://user-images.githubusercontent.com/13216011/148964646-8a9a451c-c6f5-4a80-a7ce-00374ece440c.png)

Ponieważ chcemy wiedzieć, jak nasz model sobie będzie radził, potrzebujemy móc go na czymś sprawdzić. Nie posiadamy zbioru testowego z odpowiedziami, dlatego podzielimy ten, który mamy, treningowy, na 2 części. Jednej użyjemy do trenowania, a drugiej do testowania.
 
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
```

Następnie model, którego będziemy używać. Nasza zmienna docelowa posiada 2 wartości, 0 i 1. Albo inaczej mówiąc, 2 klasy: przeżył/nie przeżył. Szukamy zatem czegoś do klasyfikacji (wynikiem jest przydział do grupy, na przykład pies/kot. W przeciwieństwie do regresji, gdzie przewidujemy wartość, na przykład cenę benzyny) i możemy wybierać między innymi między Logistic Regression, Naive Bayes, Stochastic Gradient Descent, Decision Tree, Random Forest, Support Vector Machine etc. Nie szalejmy, LogisticRegression brzmi spoko.
 
```
from sklearn.linear_model import LogisticRegression

model =  LogisticRegression()
model.fit(X_train, y_train)    # trenujemy model
```
 
I to już :)  Wiem, mało imponujące. Co tu zaszło? Algorytm dostał macierz cech X (X_train) i wektor cechy docelowej Y (y_train). Zbadał związki między jednym i drugim i zbudował funkcję, według której cechy X wpływają na wartość Y. Teraz tę funkcję chcemy zaaplikować do testowego zbioru (X_test) i wygenerować zestaw wyników. Następnie te wyniki porównamy z rzeczywistymi wynikami (y_test) i  dowiemy się, jaka jest skuteczność modelu. Żeby porównać wyniki, użyjemy accuracy_score. Jest to miara, która przedstawia liczbę wszystkich prawidłowo przewidzianych wartości w stosunku do liczby wszystkich wartości w zbiorze. Czyli jeśli prawidłowo trafiliśmy 30 wartości ze 100 elementowego zbioru, to nasza skuteczność wynosi 30%.
 
Takich miar do sprawdzania poprawności modelu jest kilka (np. Classification Accuracy, Logarithmic Loss, Confusion Matrix, Mean Squared Error etc). Jedne działają dobrze dla klasyfikacji, inne dla regresji. Z biegiem czasu człowiek uczy się intuicyjnie wybierać odpowiednie, ale zanim to nastąpi, dobrze jest mieć cheat sheet :) 
 
 ```
#sprawdzamy skuteczność modelu
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test) # używamy naszego modelu na zbiorze testowym i zapisujemy przewidziane wyniki do y_pred

accuracy_score(y_test, y_pred) # porównujemy przewidziane wyniki y_pred z rzeczywistymi wartościami w y_test
 ```
 
 W tym przypadku accuracy score to 78%. (uwaga, ta liczba zmienia się w zależności od wielkości zbioru testowego. Możemy zmienić wartość test_size w funkcji train_test_split, żeby to zaobserwować). Ten wynik nie jest super wiarygodny, pamiętajmy, że nasz wynik na kaggle może się od niego różnić o kilka procent, ponieważ będziemy testować na zupełnie innym zbiorze. Czas to sprawdzić. Przygotujmy więc dane testowe kagglowe (plik test.csv)

```
 # wczytajmy testowy zbiór Kaggle
kaggle_test = pd.read_csv('./test.csv')
```
 Musimy przygotować cechy w zbiorze testowym tak samo, jak te w treningowym. Sprawdzamy więc braki w danych i zamieniamy wartości na liczbowe tam, gdzie tego potrzebujemy.
 
 ```
 kaggle_test.loc[kaggle_test['Age'].isnull(), 'Age'] = kaggle_test['Age'].median() # uzupełnij brakujący wiek
kaggle_test.loc[kaggle_test['Fare'].isnull(), 'Fare'] = kaggle_test['Fare'].mean() # uzupełnij brakującą opłatę

# zamień literki z kolumny Embarked na cyfry
dict = {"S" : 1, "C" : 2, "Q": 3}

kaggle_test['Embarked_Int'] = kaggle_test['Embarked']
kaggle_test = kaggle_test.replace({"Embarked_Int": dict})

# zamień płcie z kolumny Sex na cyfry
kaggle_test['Sex_Int'] = kaggle_test['Sex'].apply(lambda x: 0 if x == 'female' else 1)
```
 
Przygotowujemy macierz cech i wykonujemy predykcję na zbiorze:
```
feats = [ 'Sex_Int', 'Pclass', 'Fare', 'Embarked_Int' ] #zmienne potrzebne w modelu, te same, których użyliśmy do trenowania

X_kaggle_test = kaggle_test[feats].values # macierz cech
   
pred = model.predict(X_kaggle_test) # predykcja
```

Zapisujemy do pliku submission.csv 2 kolumny: ID pasażera i nasz przewidywany wynik. Taki format pliku jest narzucony przez Kaggle, można to sprawdzić w zakładce Data. I w końcu wrzucamy nasz plik na Kaggle i sprawdzamy, jak sobie poradził :) Mój wynik to 0.76315. Myślę, że to się plasuje gdzieś w kategorii brązowego medalu, czyli nie było najgorzej, zdecydowanie może być lepiej. Ale hej, tu chodziło o zrozumienie schematu i kolejności wykonywania działań przy zwyczajnym zadaniu MLowym. Na inżynierię cech i inne cuda przyjdzie jeszcze czas.
 
PS> Btw, nie dajcie się zdeprymować wynikami równymi 1 na leaderboardzie Kaggle. To ludzie, którzy oszukiwali - [tutaj](https://www.kaggle.com/carlmcbrideellis/titanic-leaderboard-a-score-0-8-is-great) info na ten temat.
