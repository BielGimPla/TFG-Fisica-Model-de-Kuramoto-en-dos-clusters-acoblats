# Simulacions del model de Kuramoto

Aquest repositori conté el codi usat per simular el model de Kuramoto en casos mean field, amb diferents valors de N 
resultats dels quals es comparen amb resultats teòrics, i en el cas de dos clústers acoblats. 

- `simulacio_opt.py`: script principal per a dur a terme les simulacions en mean field per diferents valors de N.
- `models_opt.py`: funcions optimitzades de simulació, integració, paral·lelització de computacions
  i representació gràfica necessaries per a córrer les simulacions.
- `simulacio_acblt.py`: funcions base pel cas de dos clústers acoblats generalitzades a partir de les 
  funcions de 'models_opt.py' i la seva implementació per a córrer les simulacions. A més, també hi ha funcions que guarden els resultats obtinguts per a poder revisar-los sense haver de tornar a simular tot el sistema. Conseqüentment, també hi ha funcions de càrrega de resultats per a generar els plots a partir dels resultats guardats. 
- `plots_acblt_entregable.py`: funcions auxiliars i de generació de gràfics per als dos clústers acoblats. 

# Generar resultats del treball

Cal recordar que els resultats obtinguts depenen de les condicions inicials. Per tant, és probable que els resultats obtinguts no siguin de tot comparables amb els presentats al treball. Les comandes de `bash` per generar els resultats presentats als treballs són:
## Mean field: r(K) per N=50
```bash
python3 simulacio_opt.py --N 50 --output_dir resultats/mean_field_N50
```
## Mean field: r(K), t_c(K) i r(t) per N=10000
```bash
python3 simulacio_opt.py --N 10000 --temps --output_dir resultats/mean_field_N10000
```
## Clústers acoblats: règim estacionari
```bash
python3 simulacio_acblt_entregable.py --regim estacionari --dense_grid --K_inter_max 3 --K_inter_points 400 --N_inter_max 500 --N_inter_points 50 --repeticions 10 --output_dir resultats/clusters_estacionari
```
## Clústers acoblats: comparació temporal r(t)
```bash
python3 simulacio_acblt_entregable.py --regim no_estacionari --K_inter_values 0 0.75 1.5 3 --unions 0 125 250 500 --T_final 30 --repeticions 20 --output_dir resultats/clusters_temporal
```

## Clústers acoblats: temps de convergència en graella densa
```bash
python3 simulacio_acblt_entregable.py --regim no_estacionari --dense_grid --K_inter_max 3 --K_inter_points 501 --N_inter_max 500 --N_inter_points 501 --T_final 30 --repeticions 20 --output_dir resultats/clusters_tc
```
