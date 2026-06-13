import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker


CHECK_EVERY = 50
TC_TOL_RELAX = 4e-2
TC_TOL_SLOPE_RELAX = 6e-3
TC_WINDOW_TIME = 0.5
TC_SMOOTH_TIME = 0.25
TC_OSCILLATION_TOL = 0.12
TEMPORAL_PLOT_MAX = 10.0
TC_MAP_COLOR_MAX = 20.0


def suavitzar_corba(r_hist, h, smooth_time):
    smooth_points = int(round(float(smooth_time) / h))
    if smooth_points <= 1:
        return np.asarray(r_hist, dtype=float)

    smooth_points = min(smooth_points, len(r_hist))
    kernel = np.ones(smooth_points, dtype=float) / smooth_points
    pad_left = smooth_points // 2
    pad_right = smooth_points - 1 - pad_left
    r_pad = np.pad(np.asarray(r_hist, dtype=float), (pad_left, pad_right), mode="edge")
    return np.convolve(r_pad, kernel, mode="valid")


def detectar_t_estacionari_corba(r_hist, h, check_every=CHECK_EVERY, T_min=None,
                                 tol_relax=TC_TOL_RELAX,
                                 tol_slope_relax=TC_TOL_SLOPE_RELAX,
                                 window_time=TC_WINDOW_TIME,
                                 smooth_time=TC_SMOOTH_TIME,
                                 oscillation_tol=TC_OSCILLATION_TOL):
    r_hist = np.asarray(r_hist, dtype=float)
    if len(r_hist) == 0 or np.all(~np.isfinite(r_hist)):
        return None
    r_hist = suavitzar_corba(r_hist, h, smooth_time)

    check_every = max(1, int(check_every))
    window_time = max(h, float(window_time))
    stride = check_every
    idx = np.arange(0, len(r_hist), stride, dtype=int)
    if idx[-1] != len(r_hist) - 1:
        idx = np.append(idx, len(r_hist) - 1)

    t_sample = idx * h
    r_sample = r_hist[idx]
    valid = np.isfinite(r_sample)
    if np.sum(valid) < 3:
        return None

    if T_min is None:
        T_min = 0.0

    window_samples = max(3, int(np.ceil(window_time / (stride * h))) + 1)
    tol_slope_eff = (
        tol_slope_relax
        if tol_slope_relax is not None
        else tol_relax / max(window_time, h)
    )

    for end in range(window_samples - 1, len(r_sample)):
        start = end - window_samples + 1
        vals = r_sample[start:end + 1]
        temps = t_sample[start:end + 1]
        vals = vals[np.isfinite(vals)]
        if len(vals) < window_samples:
            continue

        t_inici = temps[0]
        if t_inici < T_min:
            continue

        amplitude = np.max(vals) - np.min(vals)
        pendent = abs(vals[-1] - vals[0]) / max(temps[-1] - temps[0], h)

        if amplitude < tol_relax and pendent < tol_slope_eff:
            post_vals = r_sample[end:]
            post_vals = post_vals[np.isfinite(post_vals)]
            if len(post_vals) == 0:
                continue
            if np.nanmax(post_vals) - np.nanmin(post_vals) > oscillation_tol:
                return None

            return float(temps[0])

    return None


def generar_plots_des_de_dades(resum, K_inter_values, unions, resultats_hist,
                               output_dir, h,
                               temporal_pairs=None,
                               generar_temporals=False,
                               regim="estacionari",
                               tol_relax=TC_TOL_RELAX,
                               check_every=CHECK_EVERY,
                               T_min=None,
                               tol_slope_relax=TC_TOL_SLOPE_RELAX,
                               tc_window_time=TC_WINDOW_TIME,
                               tc_smooth_time=TC_SMOOTH_TIME,
                               tc_oscillation_tol=TC_OSCILLATION_TOL):
    os.makedirs(output_dir, exist_ok=True)
    estudiar_convergencia = regim == "no_estacionari"

    if regim == "estacionari":
        plot_superficie_global(resum, K_inter_values, unions, output_dir)
        plot_frontera_sincronitzacio(resum, K_inter_values, unions, output_dir)
        plot_r_vs_K_Ninter_seleccionats(resum, K_inter_values, unions, output_dir)
        plot_r_vs_Ninter_Kinter_seleccionats(resum, K_inter_values, unions, output_dir)
        plot_comparacio_r1_r2_global_vs_Kinter_petits_grans(resum, K_inter_values, unions, output_dir)
        plot_comparacio_r1_r2_global_vs_Ninter_petits_grans(resum, K_inter_values, unions, output_dir)
    
    elif estudiar_convergencia:
        plot_superficie_t_convergencia(K_inter_values, unions, output_dir, resultats=resultats_hist, h=h, tol_relax=tol_relax, check_every=check_every, T_min=T_min, tol_slope_relax=tol_slope_relax, tc_window_time=tc_window_time, tc_smooth_time=tc_smooth_time, tc_oscillation_tol=tc_oscillation_tol)
        plot_mapa_t_convergencia(K_inter_values, unions, output_dir, resultats=resultats_hist, h=h, tol_relax=tol_relax, check_every=check_every, T_min=T_min, tol_slope_relax=tol_slope_relax, tc_window_time=tc_window_time, tc_smooth_time=tc_smooth_time, tc_oscillation_tol=tc_oscillation_tol)
    else:
        raise ValueError(f"Regim desconegut: {regim}")

    if generar_temporals and resultats_hist:
        plot_comparacio_temporal_rs(resultats_hist, h, output_dir, temporal_pairs=temporal_pairs, tol_relax=tol_relax, check_every=check_every, T_min=T_min, tol_slope_relax=tol_slope_relax, tc_window_time=tc_window_time, tc_smooth_time=tc_smooth_time, tc_oscillation_tol=tc_oscillation_tol, marcar_convergencia=estudiar_convergencia)
    elif generar_temporals:
        print("Sense histories r(t).")


def _graella_des_de_resum(resum, K_inter_values, unions, columna):
    Z = np.full((len(K_inter_values), len(unions)), np.nan, dtype=float)

    for i, K_inter in enumerate(K_inter_values):
        for j, n_unions in enumerate(unions):
            mask = (resum[:, 0] == K_inter) & (resum[:, 1] == n_unions)
            if np.any(mask):
                Z[i, j] = resum[mask][0, columna]

    return Z


def plot_superficie_global(resum, K_inter_values, unions, output_dir):
    X, Y = np.meshgrid(K_inter_values, unions)
    Z = _graella_des_de_resum(resum, K_inter_values, unions, columna=6)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z.T, cmap="viridis", edgecolor="0.35", linewidth=0.4, antialiased=True, alpha=0.95)

    ax.set_xlabel(r"$K_{inter}$")
    ax.set_ylabel(r"$N_{inter}$")
    ax.set_zlabel(r"$\hat{r}_{\mathrm{global}}$")
    ax.set_zlim(0, 1.05)
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.12, label=r"$\hat{r}_{\mathrm{global}}$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_superficie_global.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def _extreure_frontera_sincronitzacio(resum, K_inter_values, unions, threshold, n_consecutius=1):
    Z = _graella_des_de_resum(resum, K_inter_values, unions, columna=6)
    K_vals = np.asarray(K_inter_values, dtype=float)
    N_vals = np.asarray(unions, dtype=float)
    punts = []

    for i, K_inter in enumerate(K_vals):
        fila = Z[i]
        valid = np.isfinite(fila)
        supera = valid & (fila >= threshold)

        if n_consecutius <= 1:
            idx_supera = np.where(supera)[0]
        else:
            idx_supera = []
            for j in np.where(supera)[0]:
                j_final = j + n_consecutius
                if j_final <= len(supera) and np.all(supera[j:j_final]):
                    idx_supera.append(j)
                    break

        if len(idx_supera) == 0:
            continue

        j = idx_supera[0]
        N_crit = N_vals[j]

        punts.append((K_inter, N_crit, fila[j]))

    return np.asarray(punts, dtype=float), Z


def plot_frontera_sincronitzacio(resum, K_inter_values, unions, output_dir):
    n_consecutius = 3
    nivells = [0.60, 0.70, 0.80, 0.90, 0.95]
    _, Z = _extreure_frontera_sincronitzacio(resum, K_inter_values, unions, nivells[0], n_consecutius=n_consecutius)

    fig, ax = plt.subplots(figsize=(8, 6))
    K_grid, N_grid = np.meshgrid(K_inter_values, unions, indexing="ij")
    mapa = ax.pcolormesh(K_grid, N_grid, Z, shading="auto", cmap="viridis", vmin=0, vmax=1)

    colors_nivells = ["#ff4f8b", "#39ff14", "#00d4ff", "#ffd166", "#ff8c00"]
    for nivell, color in zip(nivells, colors_nivells):
        punts_nivell, _ = _extreure_frontera_sincronitzacio(resum, K_inter_values, unions, nivell, n_consecutius=n_consecutius)
        if len(punts_nivell) < 2:
            continue

        K_nivell = punts_nivell[:, 0]
        N_nivell = punts_nivell[:, 1]
        ordre_nivell = np.argsort(K_nivell)

        ax.plot(K_nivell[ordre_nivell], N_nivell[ordre_nivell], "--", color=color, lw=1.8, label=fr"$\hat{{r}}_{{\mathrm{{global}}}}\geq {nivell:.2f}$")

    ax.set_xlabel(r"$K_{inter}$")
    ax.set_ylabel(r"$N_{inter}$")
    ax.set_xlim(np.min(K_inter_values), np.max(K_inter_values))
    ax.set_ylim(np.min(unions), np.max(unions))
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(bbox_to_anchor=(0.5, -0.16), loc="upper center", borderaxespad=0, fontsize=9, ncol=3)
    fig.colorbar(mapa, ax=ax, label=r"$\hat{r}_{\mathrm{global}}$")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_frontera_sincronitzacio.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def _dades_per_Ninter(resum, n_unions):
    mask = resum[:, 1].astype(int) == int(n_unions)
    dades = resum[mask]
    if len(dades) == 0:
        return dades
    return dades[np.argsort(dades[:, 0])]


def _dades_per_Kinter(resum, K_inter):
    mask = np.isclose(resum[:, 0], K_inter)
    dades = resum[mask]
    if len(dades) == 0:
        return dades
    return dades[np.argsort(dades[:, 1])]


def _plot_tres_series_amb_banda(ax, x, dades, linestyle="-", alpha=0.16):
    series = [
        (2, 3, r"$\hat{r}_1$", "tab:blue"),
        (4, 5, r"$\hat{r}_2$", "tab:green"),
        (6, 7, r"$\hat{r}_{global}$", "tab:orange"),
    ]
    for col_mean, col_std, label, color in series:
        y = dades[:, col_mean]
        yerr = dades[:, col_std]
        ax.plot(x, y, linestyle=linestyle, lw=2.0, color=color, label=label)
        if np.any(np.isfinite(yerr)):
            ax.fill_between(x, np.clip(y - yerr, 0, 1.05), np.clip(y + yerr, 0, 1.05), color=color, alpha=alpha, linewidth=0)


def _plot_r_vs_variable_seleccionada(resum, K_inter_values, unions, output_dir, variable_fixa, nom_fitxer):
    K_vals = np.asarray(sorted(K_inter_values), dtype=float)
    unions_vals = np.asarray(sorted(unions), dtype=int)
    if len(K_vals) == 0 or len(unions_vals) == 0:
        return

    if variable_fixa == "N_inter":
        valors_fixats = unions_vals
        if len(valors_fixats) > 11:
            idx = np.round(np.linspace(0, len(valors_fixats) - 1, 11)).astype(int)
            valors_fixats = valors_fixats[np.unique(idx)]
        x_label = r"$K_{inter}$"
        x_limits = (float(np.min(K_vals)), float(np.max(K_vals)))
        colors = plt.cm.viridis(np.linspace(0.08, 0.98, len(valors_fixats)))
    elif variable_fixa == "K_inter":
        valors_fixats = K_vals
        if len(valors_fixats) > 11:
            idx = np.round(np.linspace(0, len(valors_fixats) - 1, 11)).astype(int)
            valors_fixats = valors_fixats[np.unique(idx)]
        x_label = r"$N_{inter}$"
        x_limits = (int(np.min(unions_vals)), int(np.max(unions_vals)))
        colors = plt.cm.plasma(np.linspace(0.08, 0.98, len(valors_fixats)))
    else:
        raise ValueError(f"variable_fixa desconeguda: {variable_fixa}")

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.225), sharex=True, sharey=True)
    camps = [
        (2, 3, r"$\hat{r}_1$"),
        (4, 5, r"$\hat{r}_2$"),
        (6, 7, r"$\hat{r}_{global}$"),
    ]

    for color, valor_fixat in zip(colors, valors_fixats):
        if variable_fixa == "N_inter":
            dades = _dades_per_Ninter(resum, int(valor_fixat))
            if len(dades) == 0:
                continue
            dades = dades[np.isin(np.round(dades[:, 0], 12), np.round(K_vals, 12))]
            x = dades[:, 0]
            label = fr"$N_{{inter}}={int(valor_fixat)}$"
            alpha = 0.10
        else:
            dades = _dades_per_Kinter(resum, float(valor_fixat))
            if len(dades) == 0:
                continue
            dades = dades[np.isin(dades[:, 1].astype(int), unions_vals)]
            x = dades[:, 1]
            K_text = f"{float(valor_fixat):.2f}".rstrip("0").rstrip(".") or "0"
            label = fr"$K_{{inter}}={K_text}$"
            alpha = 0.12
        if len(dades) == 0:
            continue
        for ax, (col_mean, col_std, ylabel) in zip(axes, camps):
            y = dades[:, col_mean]
            yerr = dades[:, col_std]
            ax.plot(x, y, color=color, lw=1.6, label=label)
            if np.any(np.isfinite(yerr)):
                ax.fill_between(x, np.clip(y - yerr, 0, 1.05), np.clip(y + yerr, 0, 1.05), color=color, alpha=alpha, linewidth=0)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)

    axes[-1].set_xlabel(x_label)
    axes[-1].set_xlim(*x_limits)
    if variable_fixa == "N_inter":
        axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{float(x):.2f}".rstrip("0").rstrip(".") or "0"))
    axes[0].legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", borderaxespad=0, fontsize=9, ncol=min(len(valors_fixats), 6), handlelength=1.8, columnspacing=0.9, handletextpad=0.4)
    plt.tight_layout(rect=(0, 0, 1, 0.99))

    fig_path = os.path.join(output_dir, nom_fitxer)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def plot_r_vs_K_Ninter_seleccionats(resum, K_inter_values, unions, output_dir):
    _plot_r_vs_variable_seleccionada(resum, K_inter_values, unions, output_dir, "N_inter", "clusters_acoblats_r_vs_K_Ninter_seleccionats.png")
    K_vals = np.asarray(sorted(K_inter_values), dtype=float)
    unions_vals = np.asarray(sorted(unions), dtype=int)
    K_detall = K_vals[K_vals <= min(0.5, float(K_vals[-1]))] if len(K_vals) > 0 else K_vals
    unions_detall = unions_vals[unions_vals <= min(100, int(unions_vals[-1]))] if len(unions_vals) > 0 else unions_vals
    _plot_r_vs_variable_seleccionada(resum, K_detall, unions_detall, output_dir, "N_inter", "clusters_acoblats_r_vs_K_Ninter_seleccionats_detall.png")
    

def plot_r_vs_Ninter_Kinter_seleccionats(resum, K_inter_values, unions, output_dir):
    _plot_r_vs_variable_seleccionada(resum, K_inter_values, unions, output_dir, "K_inter", "clusters_acoblats_r_vs_Ninter_Kinter_seleccionats.png")
    K_vals = np.asarray(sorted(K_inter_values), dtype=float)
    unions_vals = np.asarray(sorted(unions), dtype=int)
    K_detall = K_vals[K_vals <= min(0.5, float(K_vals[-1]))] if len(K_vals) > 0 else K_vals
    unions_detall = unions_vals[unions_vals <= min(100, int(unions_vals[-1]))] if len(unions_vals) > 0 else unions_vals
    _plot_r_vs_variable_seleccionada(resum, K_detall, unions_detall, output_dir, "K_inter", "clusters_acoblats_r_vs_Ninter_Kinter_seleccionats_detall.png")


def plot_comparacio_r1_r2_global_vs_Kinter_petits_grans(resum, K_inter_values,
                                                        unions, output_dir):
    unions_vals = np.asarray(sorted(unions), dtype=int)
    if len(unions_vals) == 0:
        return
    unions_plot = []
    for objectiu in [float(unions_vals[-1]) / 10.0, float(unions_vals[-1])]:
        n_unions = int(unions_vals[np.argmin(np.abs(unions_vals - objectiu))])
        if n_unions not in unions_plot:
            unions_plot.append(n_unions)

    fig, ax = plt.subplots(figsize=(5.4, 4.286))
    linestyles = ["-", ":"]
    handles_estil = []
    for idx, n_unions in enumerate(unions_plot):
        dades = _dades_per_Ninter(resum, int(round(n_unions)))
        if len(dades) == 0:
            continue
        linestyle = linestyles[min(idx, len(linestyles) - 1)]
        _plot_tres_series_amb_banda(ax, dades[:, 0], dades, linestyle=linestyle)
        handles_estil.append(plt.Line2D([0], [0], color="0.2", lw=2.0, linestyle=linestyle, label=fr"$N_{{inter}}={int(round(n_unions))}$"))

    handles_color = [
        plt.Line2D([0], [0], color="tab:blue", lw=2.0, label=r"$\hat{r}_1$"),
        plt.Line2D([0], [0], color="tab:green", lw=2.0, label=r"$\hat{r}_2$"),
        plt.Line2D([0], [0], color="tab:orange", lw=2.0, label=r"$\hat{r}_{global}$"),
    ]
    ax.legend(handles=handles_color + handles_estil, bbox_to_anchor=(0.5, 1.02), loc="lower center", borderaxespad=0, fontsize=9, ncol=len(handles_color + handles_estil), handlelength=1.6, columnspacing=0.65, handletextpad=0.35)
    ax.set_xlabel(r"$K_{inter}$")
    ax.set_ylabel(r"$\hat{r}$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(np.min(K_inter_values), np.max(K_inter_values))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{float(x):.2f}".rstrip("0").rstrip(".") or "0"))
    ax.grid(alpha=0.25)
    plt.tight_layout(rect=(0, 0, 1, 0.80))

    fig_path = os.path.join(output_dir, "clusters_acoblats_comparacio_r1_r2_global_vs_Kinter_petits_grans.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def plot_comparacio_r1_r2_global_vs_Ninter_petits_grans(resum, K_inter_values,
                                                        unions, output_dir):
    K_vals = np.asarray(sorted(K_inter_values), dtype=float)
    if len(K_vals) == 0:
        return
    K_plot = []
    for objectiu in [float(K_vals[-1]) / 10.0, float(K_vals[-1])]:
        K_inter = float(K_vals[np.argmin(np.abs(K_vals - objectiu))])
        if not any(np.isclose(K_inter, K_guardat) for K_guardat in K_plot):
            K_plot.append(K_inter)

    fig, ax = plt.subplots(figsize=(5.4, 4.286))
    linestyles = ["-", ":"]
    handles_estil = []
    for idx, K_inter in enumerate(K_plot):
        dades = _dades_per_Kinter(resum, K_inter)
        if len(dades) == 0:
            continue
        linestyle = linestyles[min(idx, len(linestyles) - 1)]
        _plot_tres_series_amb_banda(ax, dades[:, 1], dades, linestyle=linestyle)
        K_text = f"{float(K_inter):.2f}".rstrip("0").rstrip(".") or "0"
        handles_estil.append(plt.Line2D([0], [0], color="0.2", lw=2.0, linestyle=linestyle, label=fr"$K_{{inter}}={K_text}$"))

    handles_color = [
        plt.Line2D([0], [0], color="tab:blue", lw=2.0, label=r"$\hat{r}_1$"),
        plt.Line2D([0], [0], color="tab:green", lw=2.0, label=r"$\hat{r}_2$"),
        plt.Line2D([0], [0], color="tab:orange", lw=2.0, label=r"$\hat{r}_{global}$"),
    ]
    ax.legend(handles=handles_color + handles_estil, bbox_to_anchor=(0.5, 1.02), loc="lower center", borderaxespad=0, fontsize=9, ncol=len(handles_color + handles_estil), handlelength=1.6, columnspacing=0.65, handletextpad=0.35)
    ax.set_xlabel(r"$N_{inter}$")
    ax.set_ylabel(r"$\hat{r}$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(np.min(unions), np.max(unions))
    ax.grid(alpha=0.25)
    plt.tight_layout(rect=(0, 0, 1, 0.80))

    fig_path = os.path.join(output_dir, "clusters_acoblats_comparacio_r1_r2_global_vs_Ninter_petits_grans.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def _agrupar_histories_per_parella(resultats, camp):
    grups = {}
    for res in resultats:
        if camp not in res:
            continue
        clau = (res["K_inter"], res["n_unions"])
        grups.setdefault(clau, []).append(res)
    return grups


def _corba_mitjana_grup(reps, camp):
    min_len = min(len(res[camp]) for res in reps)
    if min_len == 0:
        return None
    matriu = np.vstack([res[camp][:min_len] for res in reps])
    return np.nanmean(matriu, axis=0)


def _graella_t_convergencia_des_de_mitjana(resultats, K_inter_values, unions, h,
                                           tol_relax=TC_TOL_RELAX,
                                           check_every=CHECK_EVERY,
                                           T_min=None,
                                           tol_slope_relax=TC_TOL_SLOPE_RELAX,
                                           tc_window_time=TC_WINDOW_TIME,
                                           tc_smooth_time=TC_SMOOTH_TIME,
                                           tc_oscillation_tol=TC_OSCILLATION_TOL):
    Z = np.full((len(K_inter_values), len(unions)), np.nan, dtype=float)
    grups = _agrupar_histories_per_parella(resultats, "r_global_hist")
    unions_index = {int(n_unions): j for j, n_unions in enumerate(unions)}
    K_array = np.asarray(K_inter_values, dtype=float)

    for (K_inter, n_unions), reps in grups.items():
        j = unions_index.get(int(n_unions))
        if j is None:
            continue

        idx_K = np.where(np.isclose(K_array, K_inter))[0]
        if len(idx_K) == 0:
            continue
        i = int(idx_K[0])

        corba_mitjana = _corba_mitjana_grup(reps, "r_global_hist")
        if corba_mitjana is None:
            continue

        t_est = detectar_t_estacionari_corba(corba_mitjana, h, check_every=check_every, T_min=T_min, tol_relax=tol_relax, tol_slope_relax=tol_slope_relax, window_time=tc_window_time, smooth_time=tc_smooth_time, oscillation_tol=tc_oscillation_tol)
        if t_est is not None:
            Z[i, j] = t_est

    return Z


def plot_superficie_t_convergencia(K_inter_values, unions, output_dir,
                                   resultats=None, h=None,
                                   tol_relax=TC_TOL_RELAX,
                                   check_every=CHECK_EVERY,
                                   T_min=None,
                                   tol_slope_relax=TC_TOL_SLOPE_RELAX,
                                   tc_window_time=TC_WINDOW_TIME,
                                   tc_smooth_time=TC_SMOOTH_TIME,
                                   tc_oscillation_tol=TC_OSCILLATION_TOL):
    if not resultats or h is None:
        print("Sense histories r(t).")
        return

    X, Y = np.meshgrid(unions, K_inter_values)
    Z = _graella_t_convergencia_des_de_mitjana(resultats, K_inter_values, unions, h, tol_relax=tol_relax, check_every=check_every, T_min=T_min, tol_slope_relax=tol_slope_relax, tc_window_time=tc_window_time, tc_smooth_time=tc_smooth_time, tc_oscillation_tol=tc_oscillation_tol)

    if np.all(~np.isfinite(Z)):
        print("Sense t_c finits per al plot 3D.")
        return

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    norm = mcolors.Normalize(vmin=0, vmax=TC_MAP_COLOR_MAX)

    if len(K_inter_values) >= 2 and len(unions) >= 2:
        surf = ax.plot_surface(X, Y, Z, cmap="plasma", norm=norm, edgecolor="0.35", linewidth=0.4, antialiased=True, alpha=0.95)
    else:
        surf = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=Z.ravel(), cmap="plasma", norm=norm, s=50, depthshade=True)

    ax.set_xlabel(r"$N_{\mathrm{inter}}$")
    ax.set_ylabel(r"$K_{inter}$")
    ax.set_zlabel(r"$t_c$")
    if len(unions) > 0:
        ax.set_xlim(max(unions), min(unions))
    if len(K_inter_values) > 0:
        ax.set_ylim(max(K_inter_values), min(K_inter_values))
    ax.view_init(elev=28, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.12, label=r"$t_c$", extend="max")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_superficie_t_convergencia.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def plot_mapa_t_convergencia(K_inter_values, unions, output_dir,
                             resultats=None, h=None,
                             tol_relax=TC_TOL_RELAX,
                             check_every=CHECK_EVERY,
                             T_min=None,
                             tol_slope_relax=TC_TOL_SLOPE_RELAX,
                             tc_window_time=TC_WINDOW_TIME,
                             tc_smooth_time=TC_SMOOTH_TIME,
                             tc_oscillation_tol=TC_OSCILLATION_TOL):
    if not resultats or h is None:
        print("Sense histories r(t).")
        return

    Z = _graella_t_convergencia_des_de_mitjana(resultats, K_inter_values, unions, h, tol_relax=tol_relax, check_every=check_every, T_min=T_min, tol_slope_relax=tol_slope_relax, tc_window_time=tc_window_time, tc_smooth_time=tc_smooth_time, tc_oscillation_tol=tc_oscillation_tol)

    if np.all(~np.isfinite(Z)):
        print("Sense t_c finits per al mapa 2D.")
        return

    X, Y = np.meshgrid(unions, K_inter_values)
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="0.85")

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.Normalize(vmin=0, vmax=TC_MAP_COLOR_MAX)
    mapa = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), shading="auto", cmap=cmap, norm=norm)

    ax.set_xlabel(r"$N_{\mathrm{inter}}$")
    ax.set_ylabel(r"$K_{inter}$")
    if len(unions) > 0:
        ax.set_xlim(max(unions), min(unions))
    if len(K_inter_values) > 0:
        ax.set_ylim(max(K_inter_values), min(K_inter_values))
    ax.grid(alpha=0.25)
    fig.colorbar(mapa, ax=ax, label=r"$t_c$", extend="max")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_superficie_t_convergencia_2d.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()


def _filtrar_claus_temporals(claus, temporal_pairs, nom_plot):
    claus = list(claus)
    if temporal_pairs is None:
        return claus

    claus_filtrades = []
    claus_absents = []
    for K_req, n_req in temporal_pairs:
        matches = [
            clau for clau in claus
            if clau[1] == n_req and np.isclose(clau[0], K_req)
        ]
        if matches:
            clau = sorted(matches, key=lambda x: abs(x[0] - K_req))[0]
            if clau not in claus_filtrades:
                claus_filtrades.append(clau)
        else:
            claus_absents.append((K_req, n_req))

    if claus_absents:
        text_absents = ", ".join(f"(K={K_inter:g}, N={n_unions:g})" for K_inter, n_unions in claus_absents)
        print(f"Parelles no disponibles per {nom_plot}: {text_absents}")

    return claus_filtrades


def plot_comparacio_temporal_rs(resultats, h, output_dir,
                                temporal_pairs=None, tol_relax=TC_TOL_RELAX,
                                check_every=CHECK_EVERY, T_min=None,
                                tol_slope_relax=TC_TOL_SLOPE_RELAX,
                                tc_window_time=TC_WINDOW_TIME,
                                tc_smooth_time=TC_SMOOTH_TIME,
                                tc_oscillation_tol=TC_OSCILLATION_TOL,
                                marcar_convergencia=True):
    grups = {}
    for res in resultats:
        if "r_global_hist" not in res:
            continue
        clau = (res["K_inter"], res["n_unions"])
        grups.setdefault(clau, []).append(res)

    if not grups:
        print("Sense histories r(t).")
        return

    t_plot_max = TEMPORAL_PLOT_MAX

    def t_est_mig_clau(clau):
        corba_mitjana = _corba_mitjana_grup(grups[clau], "r_global_hist")
        if corba_mitjana is None:
            return np.nan
        t_est = detectar_t_estacionari_corba(corba_mitjana, h, check_every=check_every, T_min=T_min, tol_relax=tol_relax, tol_slope_relax=tol_slope_relax, window_time=tc_window_time, smooth_time=tc_smooth_time, oscillation_tol=tc_oscillation_tol)
        return np.nan if t_est is None else t_est

    if temporal_pairs is not None:
        claus = _filtrar_claus_temporals(grups.keys(), temporal_pairs, "r(t)")
        if not claus:
            print("Sense parelles temporals.")
            return
    elif not marcar_convergencia:
        claus = sorted(grups, key=lambda x: (x[0], x[1]))
    else:
        claus_ordenades = sorted(grups, key=lambda x: (x[0], x[1]))
        claus = []
        for clau in claus_ordenades:
            t_est = t_est_mig_clau(clau)
            if np.isfinite(t_est) and t_est <= t_plot_max:
                claus.append(clau)
        for clau in claus_ordenades:
            if clau not in claus:
                claus.append(clau)

    noms_possibles = [
        ("r1_hist", r"$r_1$"),
        ("r2_hist", r"$r_2$"),
        ("r_global_hist", r"$r_{global}$"),
    ]
    noms = [
        item
        for item in noms_possibles
        if any(all(item[0] in res for res in reps) for reps in grups.values())
    ]
    if not noms:
        print("Sense histories r(t).")
        return

    fig, axes = plt.subplots(len(noms), 1, figsize=(10, 3 + 1.8 * len(noms)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    colors_base = np.array(["#D00000", "#F48C06", "#9D0208", "#FFBA08", "#E85D04", "#C1121F", "#7B2CBF", "#F77F00", "#6A040F", "#3A0CA3"])
    colors = colors_base[:len(claus)] if len(claus) <= len(colors_base) else list(colors_base) + list(plt.cm.tab20(np.linspace(0, 1, len(claus) - len(colors_base))))
    for color, clau in zip(colors, claus):
        K_inter, n_unions = clau
        reps = grups[clau]
        min_len = min(len(res["r_global_hist"]) for res in reps)
        if min_len == 0:
            continue

        t_hist = np.arange(min_len) * h
        etiqueta = fr"$K={K_inter:g}$, $N={n_unions:g}$"

        for ax, (camp, nom) in zip(axes, noms):
            if not all(camp in res for res in reps):
                continue
            matriu = np.vstack([res[camp][:min_len] for res in reps])
            mitjana = np.mean(matriu, axis=0)
            camp_std = f"{camp}_std"
            if len(reps) == 1 and camp_std in reps[0]:
                desviacio = np.asarray(reps[0][camp_std][:min_len], dtype=float)
            else:
                desviacio = np.std(matriu, axis=0)
            t_est_mig = None
            if marcar_convergencia:
                t_est_mig = detectar_t_estacionari_corba(mitjana, h, check_every=check_every, T_min=T_min, tol_relax=tol_relax, tol_slope_relax=tol_slope_relax, window_time=tc_window_time, smooth_time=tc_smooth_time, oscillation_tol=tc_oscillation_tol)

            ax.plot(t_hist, mitjana, color=color, lw=1.6, label=etiqueta)
            ax.fill_between(t_hist, np.clip(mitjana - desviacio, 0, 1.05), np.clip(mitjana + desviacio, 0, 1.05), color=color, alpha=0.14, linewidth=0)
            if t_est_mig is not None and np.isfinite(t_est_mig) and t_est_mig <= t_hist[-1]:
                r_est = np.interp(t_est_mig, t_hist, mitjana)
                ax.axvline(t_est_mig, color=color, linestyle="--", alpha=0.25, linewidth=1)
                ax.plot(t_est_mig, r_est, "o", color=color, markeredgecolor="black", markeredgewidth=0.8, markersize=8, zorder=5)
            ax.set_ylabel(nom)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Temps (s)")
    axes[-1].set_xlim(0, t_plot_max)
    axes[0].legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "clusters_acoblats_comparacio_temporal_rs.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Plot desat: {fig_path}")
    plt.close()
