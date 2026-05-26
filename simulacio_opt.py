"""
simulacio.py — Script principal de barrida de K paral·lelitzada
================================================================

Canvis respecte la versió original:
  - Els loops (P → K → repeticions) s'han reestructurat:
      · El loop sobre K és ara paral·lel (barrida_K_parallel)
      · Les repeticions s'acumulen en una llista i es mitjanen al final
      · El model (K_ij, k_mean) es construeix UNA sola vegada per P,
        no N_repeticions × N_Ks vegades com abans
  - Ruta de desament parametritzada (ja no hardcodejada)
  - Guard __main__ obligatori per a multiprocessing en Windows/macOS

Ús:
    python simulacio_opt.py --output_dir ./resultats
"""

import os
import argparse

import numpy as np
from tqdm import tqdm

from models_opt import (
    barrida_K_parallel,
    kura_mean,
    plot_comparacio_dos_osciladors,
    plot_r_vs_K,
    plot_r_vs_t,
    plot_t_convergencia_vs_K,
)


parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", type=str, default=".", help="Directori on desar el .pkl i la figura",)
parser.add_argument("--N", type=int, default=None, help="Nombre d'oscil·ladors presents a la simulació",)
parser.add_argument("--repeticions", type=int, default=20, help="Nombre de repeticions per cada valor de K",)
parser.add_argument("--temps", action="store_true", help="Estudia el temps de relaxació",)
parser.add_argument("--K_min", type=float, default=None, help="Limit inferior de K per generar una barrida densa",)
parser.add_argument("--K_max", type=float, default=None, help="Limit superior de K per generar una barrida densa",)
parser.add_argument("--K_points", type=int, default=None, help="Nombre de punts de K entre K_min i K_max",)

args = parser.parse_args()

model_type = "mean_field"
output_dir = args.output_dir
temps_hist = args.temps

os.makedirs(output_dir, exist_ok=True)


# Configuració
N = args.N
h = 0.01
T_final = 20
repeticions = args.repeticions
tol_relax = 2e-2
tol_slope_relax = 2e-3
K_convergencia_min = 2.0
T_convergencia_max = 200

def generar_Ks():
    """
    Retorna el format de K adient depenent del nombre d'oscil·ladors escollit. 
    Per N=2, selecciona una llista gairebé continua, ja que no es necessitarà molta potència de computació
    Per N>2, selecciona una llista determinada, per optimitzar el temps de computació.
    Quan es vulgui estudiar t_con(K) cadrà una llista gairebé contínua per obtenir molta resolució.
    """
    if N == 2:
        Ks = np.linspace(0.1, 5.0, 40)
    else:
        Ks = np.array([0.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 40.0])

    arguments_graella = (args.K_min, args.K_max, args.K_points)
    n_arguments_graella = sum(x is not None for x in arguments_graella)
    usar_graella = n_arguments_graella == len(arguments_graella)

    if 0 < n_arguments_graella < len(arguments_graella):
        raise ValueError("Per generar una graella cal indicar --K_min, --K_max i --K_points")
    
    if not usar_graella:
        return Ks

    K_min = float(args.K_min)
    K_max = float(args.K_max)
    K_points = args.K_points

    if K_max < K_min:
        raise ValueError("--K_max ha de ser mes gran o igual que --K_min")

    if K_points < 2:
        raise ValueError("--K_points ha de ser com a minim 2")

    return np.linspace(K_min, K_max, K_points)


def demanar_omegues():
    """
    Demana el valor de les freqüències quan N=2.
    """
    omega1 = float(input("omega1 = "))
    omega2 = float(input("omega2 = "))
    return omega1, omega2


def executar_dos_osciladors(Ks_barrida):
    omega1, omega2 = demanar_omegues()
    tqdm.write(f"Mode dos oscil·ladors: omega1={omega1:g}, omega2={omega2:g}")

    tqdm.write(f"Generant comparació teoria-simulació r(K) amb {repeticions} repeticions...")

    fig_path = os.path.join(output_dir, f"dos_osciladors_r-k_w1_{omega1:g}_w2_{omega2:g}.png")

    plot_comparacio_dos_osciladors(
            Ks_barrida,
            omega1,
            omega2,
            h=h,
            T_final=T_convergencia_max,
            output_path=fig_path,
            show_progress=True,
            repeticions=repeticions,
    )
    tqdm.write("Comparació r(K) acabada.")


def executar_mean_field(Ks_barrida):
    f = kura_mean # definició de la funció de la EDO

    r_per_rep = []   # llista de dicts {K: r_final}, un per repetició
    t_convs = {}

    tqdm.write(
            f"Barrida de {len(Ks_barrida)} valors de K amb {os.cpu_count()} workers paral·lels per repetició"
    )

    for rep in tqdm(range(repeticions), desc="Repeticions", unit="rep", dynamic_ncols=True, leave=True):

        omega = np.random.standard_cauchy(N)
        theta0 = np.random.uniform(0, 2 * np.pi, N)

        resultat_barrida = barrida_K_parallel(
                Ks_barrida, theta0, f, h, T_final, omega,
                tol=tol_relax,
                check_every=50,
                temps_hist=temps_hist,
                K_convergencia_min=K_convergencia_min,
                T_convergencia_max=T_convergencia_max,
                tol_slope=tol_slope_relax
            )
        
        # Segons el valor de temps_hist existeixen dos modes de computació.
        # Quan és fals s'estudia r(K), si és cert s'estudia r(t). Segons el cas es guarden les dadesde la manera més dient.
        if temps_hist: 
            r_finals, t_conv = resultat_barrida
            t_convs[rep]= t_conv
        else:
            r_finals = resultat_barrida

        r_per_rep.append(r_finals)
          
    r_final_mitja = []
    r_final_std = []

    for K in Ks_barrida:
        if temps_hist:
            vals = []
            for rep in range(repeticions):
                r_hist = np.asarray(r_per_rep[rep][K])
                vals.append(np.mean(r_hist[int(0.8 * len(r_hist)):]))
        else:
            vals = [r_per_rep[rep][K] for rep in range(repeticions)]

        r_final_mitja.append(np.mean(vals))
        r_final_std.append(np.std(vals))

    # Es guarden les dades en fitxers txt per a poder reproduir els plots.
    txt_path = os.path.join(output_dir, f"resultats_{model_type}_{N}.txt")
    dades = np.column_stack((Ks_barrida, r_final_mitja, r_final_std))
    np.savetxt(txt_path, dades, header="K r_final_mitja r_final_std", fmt="%.10g")
    print(f"\nResultats desats a: {txt_path}")

    # Grafica r(K)
    fig_path = os.path.join(output_dir, f"r-k_{model_type}_{N}.png")
    plot_r_vs_K(r_final_mitja, Ks_barrida, model_type, N, errors = r_final_std, output_path=fig_path)

    if temps_hist: # Per fer els càlculs i plots necessaris per estudiar r en funció del temps
        # Càlcul de la mitjana de r sobre les repeticions per a cada t i cada K.
        r_hists_mig = {}
        r_hists_std = {}

        for K in Ks_barrida:
            reps_K = [np.asarray(r_per_rep[rep][K]) for rep in range(repeticions)]
            min_len = min(len(r_hist) for r_hist in reps_K)
            matriu = np.vstack([r_hist[:min_len] for r_hist in reps_K])

            r_hists_mig[K] = np.mean(matriu, axis=0)
            r_hists_std[K] = np.std(matriu, axis=0)

        # Càlcul de la mitjana de t_conv sobre repeticions per a cada K
        t_conv_mig = []
        t_conv_std = []

        for K in Ks_barrida:
            if K <= K_convergencia_min: # Per a casos de K<Kc
                t_conv_mig.append(np.nan)
                t_conv_std.append(np.nan)
                continue

            vals = [t_convs[rep][K] for rep in range(repeticions) if t_convs[rep][K] is not None]
            t_conv_mig.append(np.mean(vals) if vals else np.nan)
            t_conv_std.append(np.std(vals) if vals else np.nan)

        # Valors mitjans de t_c que s'escriuran a la llegenda del plot r(t).
        t_conv_plot = {}
        for i, K in enumerate(Ks_barrida):
            if K > K_convergencia_min and np.isfinite(t_conv_mig[i]):
                t_conv_plot[K] = t_conv_mig[i]
        
        # Es guarden els resultats per a poder reproduir els plots
        txt_path = os.path.join(output_dir, f"t_conv_{model_type}_{N}.txt")
        dades = np.column_stack((Ks_barrida, t_conv_mig, t_conv_std))
        np.savetxt(txt_path, dades, header="K t_conv_mig t_conv_std", fmt="%.10g")
        print(f"\nResultats desats a: {txt_path}")
        
        # Grafica t_conv(K)
        tconv_fig_path = os.path.join(output_dir, f"t_conv-k_{model_type}_{N}.png")
        plot_t_convergencia_vs_K(
                Ks_barrida,
                t_conv_mig,
                t_conv_std=t_conv_std,
                output_path=tconv_fig_path,
                model_type=model_type,
                N=N,
                K_convergencia_min=K_convergencia_min,
        )
        
        # Grafica r(t), nomes si hi ha pocs K per mantenir el plot llegible
        if len(Ks_barrida) <= 20:
            fig_path = os.path.join(output_dir, f"r-t_{model_type}_{N}.png")
            plot_r_vs_t(r_hists_mig, model_type, N,
                        output_path=fig_path,
                        t_convs=t_conv_plot,
                        T_plot=T_final,
                        K_convergencia_min=K_convergencia_min,
                        r_hists_std=r_hists_std,
                        h=h)
        else:
            print("S'omet el plot r(t) perque la barrida te mes de 20 valors de K.")


# Bloc Principal
def main():
    Ks_barrida = generar_Ks()

    if N == 2:
        executar_dos_osciladors(Ks_barrida)
        return

    executar_mean_field(Ks_barrida)
    

if __name__ == "__main__":
    main()
