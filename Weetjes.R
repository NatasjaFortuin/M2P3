#WEETJES----
#gebruik deze regel met de hashtag om aan te geven wat je probeert te coderen. Werkt als een log voor review. 
#maak regels script ed. niet langer dan 80 karakters lang. Dit zijn er zoveel..
#alt-enter to use to execute a command (call) in console voor alle bovenliggende regels
#ctrl-enter to run one line/call
#F1 ?sqrt is help functie binnen r. Hiervoor moet je cursor op het woord staan
#rm staat voor remove rm(cars) verwijdert het databestand cars (je kunt geen bestanden van R zelf weggooien). rm(list=ls()) is bezem en schoont alle bestanden uit je environment op. 
#PLUS TEKEN +
#bij een incomplete call geeft R een + terug op de volgende regel. Dan opnieuw beginnen of call aanvullen. 
#Functie Glimpse in plyr geeft een tussenstand van de calls.
#Iedere rij in je data is een observatie
#CTRL SHIFT N daarmee open je een nieuwe leeg script
#data(mtcars) commando (noemen ze CALL in R) geeft aan welk databestand van toepassing is voor het commando
#datasummary geeft mean/median etc. summarize() 
#mijn_sel (is je eigen call/selectie) <- select(mtcars, "miles per gallon" = mpg, "motor"  = disp, starts_with())
#mijnfilter <- (geef je het een eigen naam/functie) filter(mtcars, mpg >hp) kun je mee filteren
#rm(a)om een object genaamd a uit de Global Environment te verwijderen
#rm(name) om een object genaamd name uit de Global Environment te verwijderen
#remove : list = list objects
#ctrl-enter geeft run commando voor 1 regel
#alt-enter geeft run commando voor alle regels in je script
#Je gebruikt ‘labs’ om verschillende argumenten te combineren

#CUT----
#cut(x=data$column, breaks = 5) R creates 5 groups within the column to make it
#more readable (ordering from min/max and then slit it in 5 groups)

#VARIABELE----
#Aanmaken van variabelen doe je met <- of met =

#OPSOMMING----
100:130 #geeft een opsomming van alle getallen tussen 100 en 130 en laten tussen brackets zien hoeveel het er zijn (31)

#DUBBELEPUNT----
# dubbele punt : is een colon operator. En geeft elke integer aan tussen twee getallen.
#Je kunt zien welke functies er zitten in een pacage door het bestand in te voeren met :: bijv. Readr:: 

#COMMENTAAR----
#ctrl-shift-r maakt het mogelijk om een nieuwe sectie te beginnen
#--- maakt een label aan dat rechtsboven een quick reference aanmaakt

#ANNULEREN----
# om een command te annuleren (omdat het erg lang duurt) gebruik je ctrl+c

#VECTOR----
#het resultaat dat r geeft van 100:130 is een serie integere getallen tussen 100 en 130. Deze opsomming heet een vector.
#alle waarden in r heten vectoren. Als je een object aanmaakt naam <- "natasja" dan is iedere letter in de naam een waarde.
#A vector is a sequence of the same **data type.**

#SELECT COLUMN----
#nfexercise$SL selecteert een kolom
#nfexercise["SL"] selecteert ook een kolom
#nfexercise[,1] selecteert ook een kolom
#als je iets tussen [] plaatst dan is is de notatie row,kolom. dus [,4] is 4e kolom

#SELECT ROW----
#nfexercise[1,1] = [row,kolom]

#OBJECT----
#als je dit resultaat van een code wilt bewaren voor gebruik later dan moet je er een object van maken. Dit stored data
#een object kan van alles zijn. Een formule, een data frame, tabel, lijst(lists) en vectoren
#rijtje <- c(1,3,5,7) maakt een object dat rechts in environment verschijnt
#ander_rijtje <- seq(from = 1, to = 10, by = 0.5) maakt een object aan dat rechts in environment verschijnt
#random_rijtje <- runif(n = 10) maakt een object aan dat rechts in environment verschijnt
#om een object te maken kies je een (elke) naam en dit volg je op met <- en een waarde of opdracht die je er aan geeft.
#iedere keer erna dat je de naam van het object invoert vervangt r dit met de waarde. a <- 1 is dan a =1 en verschijnt rechts ->
#de environment pane is dan ook een archief van alle objecten die je hebt aangemaakt (startnaam niet met nummers of tekens!) 
#de namen van objecten zijn hoofdletter gevoelig. R en r zijn dus verschillende objecten. Als je ze vaker gebruikt overschrijft het!
#als je ls() invoert kun je zien welke objecten je allemaal al hebt gemaakt. Dit voorkomt ook dat je al bestaande objecten overschrijft.

#IMPORT DATA----
#Importing "From Text (readr)" files allows you to import CSV files and in general, 
#character delimited files using the readr package. Je kunt zo ook urls importeren!
#This Text importer provides support to:
#Import from the file system or a url
#Change column data types
#Skip or include-only columns
#Rename the data set
#Skip the first N rows
#Use the header row for column names
#Trim spaces in names
#Change the column delimiter
#Encoding selection
#Select quote, escape, comment and NA identifiers
#MANUAL CODE: nfcars <- read.csv("C:\\Users\\NFortuin\\Desktop\\Ubiqum\\Data\\Module 2\\P1\\nfcars.csv")

#SELECT----
#select(1:3) #geeft aan welke kolommen je selecteert, in dit geval 1 tot 3

#DATATYPE WIJZIGEN---- 
#link voor meer: https://www.statmethods.net/management/typeconversion.html
#is.numeric(), is.character(), is.vector(), is.matrix(), is.data.frame()
#as.numeric(), as.character(), as.vector(), as.matrix(), as.data.frame)

#MISSING VALUES----
#is.na() identifies missing values
#sum(is.na(nfexercise)) gives # NA's
#mean(is.na(nfexercise)) gives the mean
#median(nfexercise$PW, na.rm = TRUE) rekent en definieert median en laat NA erbuiten
#nfexercise$PW <- ifelse(is.na(nfexercise$PW),median(nfexercise$PL,na.rm = TRUE),nfexercise$PW)
 #met deze voorgaande call (ifelse) VERVANG je NA door de median en laat andere waarden hun eigen waarde houden
#na.omit() deletes incomplete observations
#maak wel altijd een nieuw object aan met <- om de kolom ex NA te overschrijven bijv:
#Soort <- na.omit(nfexercise$Soort)
#als we bijv de variabele leeftijd willen veranderen van numeric naar character, gebruik dan as.character-functie.
#als je een kolom van "factor" in character wil veranderen dan gebruik je: as.character(nfcars$name) of 
#as.factor(nfcars$name) als je de kolom names wilt omzetten van character naar factor
#is.na(nfcars) geeft overzicht of er missing data is. TRUE is ja, FALSE = nee
#Remove any observations containing missing data (If the missing data is less than 10% of the total data 
#and only after comparing the min/max of all the features both with and without the missing data.)
#na.omit(nfcars$ColumnName)
#Drops any rows with missing values and omits them forever.
#ook mogelijk: Replace the missing values with the mean, which is common technique, 
#but something to use with care with as it can skew the data.
#DatasetName$ColumnName[is.na(DatasetName$ColumnName)]<-mean(DatasetName$ColumnName,na.rm = TRUE)
#link voor meer: https://stats.idre.ucla.edu/r/faq/how-does-r-handle-missing-values/

 #MEANMEDIAN----
 #median(nfexercise$SL, na.rm = TRUE) rekent median uit in data/kolom ex NA's
 
#DATATYPES----
#Here are the data types that you will encounter when working in R:
#Numeric- Numbers with decimals. (Ex: 1.0, 10.5, 4.5, etc.)
#Integer Data- Whole numbers (Ex: 11, 45, 78, etc.)
#Factor Data- Categorical data (Ex: Red, Blue, Green, Purple, etc.)
#Ordinal Data- Ordered data (Ex: educational levels, temperature, etc.)
#Character Data- String values, which are characters (words) with quotes around them. (Ex: "Red", "Blue", "Green", "Purple", etc.)
#Logical- TRUE or FALSE (Always capitalize TRUE or FALSE)
#TIP: Some common data transitions are numeric to factor, character to factor, integer to numeric.
#meer over datatypes: http://www.cyclismo.org/tutorial/R/types.html

#OUTLIERS----
#nfexercise$PW[which(nfexercise$PW>401)] (met min/max functie eerst outlier gezocht = 401)
#outliers <- boxplot(nfexercise$PW, plot=FALSE)$out check en laat in boxplot de outlier zien
#print(outliers) Laat na voorgaande calls de gedetecteerde outlier zien
#nfexercise[which(nfexercise$PW %in% outliers),]
#nfexercise <- nfexercise[-which(nfexercise$PW %in% outliers),]
#boxplot(nfexercise$PW)

#KOLOMNAAM WIJZIGEN----
#names(nfcars) <-c("name", "speed", "distance")

#WORKING DIRECTORY----
#De Working Directory is een belangrijk onderdeel in R. Het is namelijk de map op jouw machine waar
#de R-sessie op zoek gaat naar bestanden als deze geïmporteerd moeten worden of waar R bestanden naar
#exporteert, bijvoorbeeld databestanden of andere r scripts. In programmeertermen kun je zeggen dat de
#Working Directory een ander woord is voor de root directory van de huidige R-sessie. Je kunt het getwd()
#command gebruiken om te kijken wat de working directory voor de huidige R-sessie is. Met het setwd()
#command kun je de working directory veranderen.

#KETEN OPERATORS----
#voert een keten van operators (zijnde opdrachten) achter elkaar uit tot je afsluit met pipe
#data %>% command voert achtereenvolgens de ingevoerde functies als call uit tussen %>%
 select(mpg, disp, hp) %>% 
  filter(mpg>21) %>% 
  mutate(mpgsq = mpg^2) %>% 
            
#NAAMWIJZIGEN----
#getwd() #setwd() kun je gebruiken om de naam te veranderen

#FUNCTIE----
#en functie is een reeks aan handelingen en bewerkingen op een waarde in R. Net zoals bij andere programmeertalen
#zijn functies een van de belangrijkste bouwstenen van R. R komt standaard met een breed scala
#aan functies waarmee je uitgebreid data kunt analyseren. Daarbij kun je packages installeren die andere
#functies bevatten die R niet heeft. Deze packages zijn verzamelingen van functies die andere personen gemaakt
#hebben en beschikbaar stellen.Een voorbeeld van een functie is: round (3.145) = afronden of factorial (3) is vermenigvuldig
#args(...) functienaam geeft aan uit welke argumenten de functie bestaat
#zelf een functie maken doe je dmv braces mijn_functie() <- {meerdere regels van code die je afsluit met}

#PRINT---- 
#met de functie print kun je R een output terug laten geven
#print ("aanhalingstekens geven aan tot hoe ver dit gaat :)"). Tekst staat dus altijd tussen haakjes "...."

#LIBRARY----
#een library laad je met het volgende call: library(“…..”).

#TRUE / FALSE EN |
#True = 1 en False = 0 in R. | is OR

#C (c) command ----
#GEBRUIK VAN KLEINE LETTER c
#c als c voor een vector staat betekent dit dat je een opsomming geeft oftewel een verzameling die je wilt combineren
#Een vector kan ook meerdere waarden bevatten. Om meerdere waarden toe te wijzen aan een vector, gebruik
#je de c() functie. vector <- (waarden, tussen, de, haakjes, scheidt, je, door, komma). Vector - (..., ..., ...,...) 

#MUTATE----
#mutate() kun je kolommen mee toevoegen of overschrijven

#PACKAGES----
#handige packages zijn tibble om een tabel te maken en ggplot om een visual te maken zoals bijvoorbeeld een lineaire lijn
#library(tibble) # package om een tabel te maken
#library(ggplot2) # package om een grafiek, histogram, lineaire lijn e.d. te maken

#TIBBLE----
# create a table in R and name it "auto". DEZE OPDRACHT kun je als volgt 'programmeren' met onderstaand command (zonder #'s)
#auto <- tibble(
  #snelheid = c(33.0, 33.0, 49.1, 65.2, 78.5, 93.0),
  #remweg = c(4.69, 4.05, 10.3, 22.3, 34.4, 43.5))

#GGPLOT----
#(+teken in ggplot staat voor EN)
# plot the data and draw a line. . DEZE OPDRACHT kun je als volgt 'programmeren' met onderstaand command (zonder #'s) 
#ggplot(data = auto, aes(x = snelheid, y = remweg)) +
  #geom_point() +
  #geom_smooth(method = "lm", se = FALSE)
#model <- lm(remweg ~ snelheid, data = auto)
#principe is steeds dat je eerst zegt welke data je wilt gebruiken (DATA) + welk model (geom_boxplot) en hoe je die wilt inrichten (aes(x…, y…., welke kolom dmv fill…). 

#LINEAR MODEL----
#in ggplot saat (lm) voor lineair model. Met de factor kun je ook een tweede kolom/observatie toevoegen aan een grafiek. 
#Als je meerdere kolom/observaties wilt gebruiken in een grafiek dan gebruik je een factor. 
#Hierbij gebruik je het argument color om de kolom cyl (dmv een kleur) toe te voegen.

 ggplot(mtcars) +
  geom_boxplot(aes(x = gear, y = mpg, fill = gear)) +
  geom_jitter(aes(x = gear, y = mpg, color = cyl), width = 0.1)
#voor een lineaire lijn (nfcars) is call: 
 library(ggplot2)
 library(dplyr)
 ggplot(nfcars,aes(x =speed,y =distance))+ 
   geom_smooth(method='lm')

#SUMMARY----
#als je summary(databestand) invoert in console dan rekent die automatisch de mean/median etc. uit
#summarise(mpg = mean(mpg)

#DATAEXPLORATIE----
 #attributes(U_cars)
 #summary(U_cars) berekent de belangrijkste statistische kengetallen zoals mean, median, quartiles
 #str(U_cars) laat de structuur van de dataset zien
 #names(U-cars)
 #dim(U_cars) laat de dimensies van de dataset zien (aantal rijen en kolommen)
 #nrow(U_cars) laat het aantal rijen in de dataset zien
 #ncol(U_cars) laat het aantal kolommen in de dataset zien
 #colnames(U_cars) laat de kolomnamen van de dataset weergeven
 #summary(U_cars$`speed of car`) laat een enkele variabele zien
 
 #REMOVE/EXCL VALUES----
 # met - (min teken) verwijder je waarden uit je functie
 # met ! met het uitroepteken kun je gegeven uitsluiten obv TRUE/FALSE
 #dit gebruik je als je een selectie maakt van specifieke [r,k] of waarden boven >..

#TESTTRAIN---- 
 #link voor meer: https://stats.stackexchange.com/questions/86285/random-number-set-seedn-in-r
 #set.seed(...) vaak getal 123 tussen haakjes om een test en train set te creeeren
 #gebruikelijk is een splitsing van 70/30 (of80/20). 70% trainingsdata en 30% testomvang binnen je data.
 #trainSize<-round(nrow(DatasetName)*0.7) call om data te splitten. Maakt nog GEEN datasets aan!
 #testSize<-nrow(DatasetName)-trainSize commando om data te splitten. Maakt nog GEEN datasets aan!
 #voer trainSize of testSize in om te controleren uit hoeveel instances je set bestaat
 #AANMAKEN VAN DE TRAIN EN TEST SETS
 #training_indices<-sample(seq_len(nrow(DatasetName)),size =trainSize)
 #trainSet<-DatasetName[training_indices,]
 #testSet<-DatasetName[-training_indices,] 
 
 #LINEAR_REG MODEL----
 #meer: http://www.endmemo.com/program/R/lm.php & https://stats.stackexchange.com/questions/5135/interpretation-of-rs-lm-output
 #Linear reg is helpful when to discover relationship between two (x,y) variables
 #x = predictor value or the indepent variable. Y = response variable or dependent value
 #geefeennaamvandevoorspelling...bijv eennaam <- lm(distance~ speed, trainSet)
 #met het commando: summary(eennaam)
 #call to create a lm:
 #set.seed(123) = random number generation to create a sample set 
 #trainSize <- round(nrow(nfiris)*0.2) = rounds numeric in rows (nrow) to two decimals
 #testSize <-nrow(nfiris)-trainSize = create a testset base on the trainingset
 #trainSize = shows the size
 #testSize = show the size
 #trainSet<-nfiris[training_indices,] =train outcome based on index (values) in rows
 #testSet<-nfiris[training_indices,] =test outcome based on index (values) in rows
 #PredictPetal.Length <- (lm(Petal.Width~ Petal.Length, trainSet)) = predicts outcome y based on x from trainSet
 #summary(PredictPetal.Length) 
 #predictions<-predict(PredictPetal.Length) =gives a lists/table of the outcome(predictions)
 #predictions 
 
 #RESIDUALS----
 #geeft 5 punten obv mean=0 en hiermee kun je snel zien of er grote outliners zijn
 #ook kun je zien of er een afwijkende distributie is uitgaande van normaal verdeling
 #residual standard error 
 
 #COEFFICIENTS----
 #geeft iets aan over de standaard afwijking
 #signif. codes zegt iets over significantie
 
 #P-VALUE---- 
 #meer interpretatie over R's lm output: https://stats.stackexchange.com/questions/5135/interpretation-of-rs-lm-output
 #p-value tells you how much the Independent Variable/Predictor affects the Dependent Variable/Response 
 #a p-value of more than 0.05 means the Independent Variable has no effect on the Dependent Variable 
 #less than 0.05 means the relationship is statistically significant.
 
 #MULTIPLE RSQUARED----
 #moet zo hoog mogelijk zijn en rond de 1. Zegt iets over hoe de lijn loopt (pos recht omhoog etc.)
 
 #PREDICT----
 #to predict y f.e. distances via x f.e. speed. Use prediction function – predict() 
#PredictionsName <- predict(TrainedModelName,testSet)
 #op basis van het maken van de trainingset onder Linear reg: voorspelspeed <- lm(distance~ speed, trainSet)
 #kun je distance laten voorspellen. De call hiervoor is:
 #voorspeldistance <- predict(voorspelspeed,testSet)
 #met call voorspeldistance krijg je dan de uitkomsten van de voorspelde data
 
 #SAMPLE---- 
 #meer: https://www.dummies.com/programming/r/how-to-take-samples-from-data-in-r/
 
 #HISTOGRAM----
 #call: hist(nfiris$Soort) =hist(data$kolomnaam)
 
 #HEATMAP
 #install.packages("corrplot")
 #corrplot::corrplot(cor(nfexercise[,1:4]))
 
 #QQNORM----
 #qqnorm(nfiris$Sepal.Width, pch = 1, frame = FALSE)
 #qqline(nfiris$Sepal.Width, col = "steelblue", lwd = 2)
 
 #...~.TOTI dot----
 #I want to model .... as a function of everyting else in the dataset
 
 #CENTER, SCALE----
 #center,scale is feitelijk een normalization opdracht als je waarden een grote range hebben
 #zoals de age en salary waarden in Blackwell M2P2 opdracht. Door in het train model als laatste
 #regel center,scale op te nemen normaliseer je de waarden tussen 0 en 1 en maak je ze vergelijkbaar

 #SAVE AS .RDS----
 #saveRDS(ModelName, file = "rfFit.rds")
 #Dit bestandje kun je vervolgens in Markdown zetten als referentie;
 # model <- readRDS("rfFit.rds")
 
#MARKDOWN----
 #laad eerst alle packages en data die je hebt gebruikt
 #sla modellen op met .rds
 #en bouw dan je rapport op
 
 #KABLEMARKDOWN----
 #library kableExtra
 #als je een dataframe in je rapport wilt opnemen gebruik dan dit script
# kable(head(exampledata)) %>% 
#   +   kable_styling()