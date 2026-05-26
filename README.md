# Simulacions del model de Kuramoto

Aquest repositori conté codi per simular el model de Kuramoto en casos mean field, amb diferents valors de N 
resultats dels quals es comparen amb resultats teòrics, i en el cas de dos clústers acoblats. 

- `simulacio_opt.py`: script principal per a dur a terme les simulacions per diferents valors de N.
- `models_opt.py`: funcions optimitzades de simulació, integració, paral·lelització de computacions
  i representació gràfica necessaries per a córrer les simulacions.
- `simulacio_acblt.py`: funcions base pel cas de dos clústers acoblats generalitzades a partir de les 
  funcions de 'models_opt.py' i la seva implementació per a córrer les simulacions. 

## Requisits

El codi està escrit en Python i utilitza:

```bash
numpy
matplotlib
os
argparse
tqdm
mpl_toolkits.axes_grid1.inset_locator
multiprocessing
```

## Model mean-field

Per a la majoria de resultats s'utilitza el model de Kuramoto globalment
connectat:

```math
\dot{\theta_i}
=
\omega_i
+
\frac{K}{N}
\sum_{j=1}^{N}
\sin(\theta_j-\theta_i).
```

En el codi no es construeix explícitament la matriu completa d'acoblament.
S'utilitza la forma mean-field equivalent:

```math
\dot{\theta_i}
=
\omega_i
+
K r \sin(\psi-\theta_i),
```

on

```math
r e^{i\psi}
=
\frac{1}{N}
\sum_{j=1}^{N} e^{i\theta_j}.
```

Això redueix el cost computacional de cada pas temporal i permet simular valors
grans de `N`, com `N = 10000`, sense haver de guardar una matriu `N x N`.

Les freqüències naturals es generen amb una distribució de Cauchy estàndard,
és a dir, amb amplada `gamma = 1`. En aquest cas, la teoria mean-field prediu
una transició crítica a

```math
K_c = 2\gamma = 2.
```

Per sobre del punt crític, la solució estacionària teòrica emprada als gràfics és:

```math
r =
\sqrt{1-\frac{K_c}{K}},
\qquad K > K_c.
```

## Integració numèrica

Les equacions diferencials s'integren amb Runge-Kutta clàssic d'ordre 4
(`rk4`) i pas temporal fix:

```python
h = 0.01
T_final = 20
```

Per cada valor de `K`, el codi calcula l'evolució temporal de `r(t)`. Quan només
s'estudia `r(K)`, el valor estacionari es calcula com la mitjana temporal del
20% final de la simulació.

Les repeticions es fan amb condicions inicials i freqüències naturals aleatòries
diferents. Els punts dels gràfics representen la mitjana sobre repeticions i les
barres d'error representen la desviació estàndard.

## Ús de `simulacio_opt.py`

La forma general és:

```bash
python simulacio_opt.py --N <nombre_oscil·ladors> --repeticions <n> --output_dir <directori>
```

Arguments principals:

- `--N`: nombre d'oscil·ladors.
- `--repeticions`: nombre de realitzacions independents per cada valor de `K`.
- `--output_dir`: directori on es desen els fitxers `.txt` i les figures.
- `--temps`: activa l'estudi temporal `r(t)` i el càlcul del temps de convergència.
- `--K_min`, `--K_max`, `--K_points`: permeten fer una graella densa de valors de `K`.

Si no s'indica una graella manual, el codi utilitza:

```python
Ks = [0.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0]
```

per a `N > 2`, i una graella més densa entre `0.1` i `5.0` per al cas de dos
oscil·ladors.

## Cas de dos oscil·ladors

Quan `N = 2`, el programa demana interactivament les freqüències `omega1` i
`omega2`. El cas mostrat als resultats correspon a:

```math
\omega_1 = 1,
\qquad
\omega_2 = 0.
```

Es pot reproduir amb:

```bash
printf "1\n0\n" | python simulacio_opt.py --N 2 --repeticions 20 --output_dir resultats_dos_osciladors
```

El programa genera una figura del tipus:

```text
dos_osciladors_r-k_w1_1_w2_0.png
```

Per a dos oscil·ladors, el llindar teòric és:

```math
K_c = |\omega_1-\omega_2|.
```

Quan `K >= K_c`, els oscil·ladors poden quedar bloquejats en fase. La corba
teòrica que es compara amb la simulació és:

```math
r = \cos\left(\frac{\Delta}{2}\right),
\qquad
\Delta = \arcsin\left(\frac{\omega_1-\omega_2}{K}\right).
```

La figura `dos_osciladors_r-k_w1_1_w2_0.png` mostra que la simulació segueix bé
la branca teòrica per sobre de `K_c = 1`. Per sota del llindar no hi ha bloqueig
estacionari entre els dos oscil·ladors, i el valor mitjà de `r` reflecteix el
moviment relatiu persistent entre les dues fases.

## Corbes `r(K)` per al model mean-field

Per generar la corba de sincronització amb `N = 10`:

```bash
python simulacio_opt.py --N 10 --repeticions 20 --output_dir resultats_N10
```

La figura generada és:

```text
r-k_mean_field_10.png
```

Per al cas gran:

```bash
python simulacio_opt.py --N 10000 --repeticions 20 --output_dir resultats_N10000
```

La figura generada és:

```text
r-k_mean_field_10000.png
```


Els valors numèrics utilitzats per construir aquests gràfics es desen en:

```text
resultats_mean_field_<N>.txt
```

amb les columnes:

```text
K r_final_mitja r_final_std
```

Per tal de no haver de torna a fer les simulacions en cas que es vulgui canviar les gràfiques.

## Evolució temporal `r(t)`

Per estudiar l'evolució temporal del paràmetre d'ordre i el temps de relaxació:

```bash
python simulacio_opt.py --N 10000 --repeticions 20 --temps --output_dir resultats_temps_N10000
```

Això genera, entre altres sortides:

```text
r-t_mean_field_10000.png
t_conv-k_mean_field_10000.png
```

En aquest mode, el codi també estima un temps de convergència `t_c`. El criteri
utilitzat busca una finestra temporal on:

- l'amplitud de les oscil·lacions de `r(t)` sigui menor que `tol_relax`;
- el pendent efectiu de `r(t)` sigui menor que `tol_slope_relax`.

Els valors actuals són:

```python
tol_relax = 2e-2
tol_slope_relax = 2e-3
K_convergencia_min = 2.0
T_convergencia_max = 200
```

Per això només s'assigna `t_c` als valors de `K` per sobre del llindar crític.

Els resultats numèrics del temps de convergència es desen en:

```text
t_conv_mean_field_<N>.txt
```

amb les columnes:

```text
K t_conv_mig t_conv_std
```

## Corba densa de temps de convergència

Per obtenir una figura més resolta de `t_c(K)`, com la versió replotada
`t_conv-k_mean_field_10000_replot.png`, es pot executar una graella densa:

```bash
python simulacio_opt.py \
  --N 10000 \
  --repeticions 20 \
  --temps \
  --K_min 2.0 \
  --K_max 15.0 \
  --K_points 150 \
  --output_dir resultats_tconv_N10000
```

Quan la graella té molts valors de `K`, el codi omet el gràfic `r(t)` per evitar
una figura massa carregada, però continua generant el fitxer de dades i el gràfic
del temps de convergència. El nom generat pel script és
`t_conv-k_mean_field_10000.png`; la versió amb sufix `_replot` correspon a una
representació posterior de les mateixes dades.


## Notes sobre reproductibilitat

Els resultats són estadístics: cada execució genera noves freqüències naturals i
noves condicions inicials. Per aquest motiu, les figures poden no ser
exactament idèntiques entre execucions, especialment per `N` petit o prop de
`K_c`.

Els fitxers `.txt` generats pel codi permeten conservar les dades utilitzades
per cada figura i tornar-les a representar sense haver de repetir tota la
simulació.
