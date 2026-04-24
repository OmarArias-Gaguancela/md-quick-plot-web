# analyzer.py — Clean server-side version of MDAnalyzer
# Stripped of all Google Colab / IPython / widget dependencies

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.rms import RMSF
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10


def _to_sci(value: float, decimals: int = 2) -> str:
    """Format a float in scientific notation with Unicode superscripts.
    e.g. -12345.67 → '-1.23 × 10⁴'
    """
    formatted = f"{value:.{decimals}e}"
    mantissa, exp = formatted.split("e")
    exp_int = int(exp)
    superscript_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp_str = str(exp_int).translate(superscript_map)
    return f"{mantissa} × 10{exp_str}"


class MDAnalyzer:
    def __init__(self, topology, trajectory,
                 protein_selection="protein",
                 ligand_selection="resname LIG",
                 dt_in_ps=None):
        if dt_in_ps is not None:
            self.u = mda.Universe(topology, trajectory, dt=dt_in_ps)
        else:
            self.u = mda.Universe(topology, trajectory)

        self.protein = self.u.select_atoms(protein_selection)
        self.ligand = self.u.select_atoms(ligand_selection)
        self.complex = self.u.select_atoms(f"{protein_selection} or {ligand_selection}")
        self.n_frames = len(self.u.trajectory)
        self.dt = self.u.trajectory.dt

    def info(self):
        return {
            "n_frames": int(self.n_frames),
            "dt_ps": float(self.dt),
            "total_time_ns": float(self.n_frames * self.dt / 1000.0),
            "protein_atoms": int(len(self.protein)),
            "ligand_atoms": int(len(self.ligand)),
            "has_ligand": len(self.ligand) > 0,
        }

    # ------------------------------------------------------------------ RMSD
    def calculate_rmsd(self, selection="backbone", reference_frame=0):
        if selection == "backbone":
            sel = self.protein.select_atoms("backbone")
        elif selection == "ca":
            sel = self.protein.select_atoms("name CA")
        else:
            sel = self.protein.select_atoms(selection)

        R = rms.RMSD(sel, sel, select=selection, ref_frame=reference_frame)
        R.run()
        time_ns = R.results.rmsd[:, 1] / 1000.0
        rmsd_values = R.results.rmsd[:, 2]
        return time_ns, rmsd_values

    def plot_rmsd(self, save_path="rmsd.png"):
        time_ns, rmsd_values = self.calculate_rmsd()
        mean_rmsd = float(np.mean(rmsd_values))
        std_rmsd = float(np.std(rmsd_values))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ns, rmsd_values, linewidth=1.5, color="#2E86AB")
        ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSD (Å)", fontsize=12, fontweight="bold")
        ax.set_title("Backbone RMSD over Time", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(mean_rmsd, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_rmsd:.2f} Å")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- CSV export ---
        csv_path = save_path.replace(".png", ".csv")
        pd.DataFrame({
            "Frame":          np.arange(len(time_ns)),
            "Time_ns":        np.round(time_ns, 4),
            "RMSD_Angstrom":  np.round(rmsd_values, 4),
        }).to_csv(csv_path, index=False)

        return {
            "plot": save_path,
            "csv":  csv_path,
            "mean": round(mean_rmsd, 3),
            "std":  round(std_rmsd, 3),
            "unit": "Å",
        }

    # ------------------------------------------------------------------ RMSF
    def calculate_rmsf(self):
        ca_atoms = self.protein.select_atoms("name CA")
        align.AlignTraj(self.u, self.u, select="backbone", in_memory=True).run()
        rmsf_obj = RMSF(ca_atoms).run()
        return ca_atoms.resnums, rmsf_obj.results.rmsf

    def plot_rmsf(self, save_path="rmsf.png"):
        residues, rmsf_values = self.calculate_rmsf()
        mean_rmsf = float(np.mean(rmsf_values))
        std_rmsf = float(np.std(rmsf_values))
        threshold = mean_rmsf + std_rmsf
        high_fluct = rmsf_values > threshold
        max_res = int(residues[np.argmax(rmsf_values)])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(residues, rmsf_values, linewidth=1.5, color="#A23B72")
        ax.set_xlabel("Residue Number", fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSF (Å)", fontsize=12, fontweight="bold")
        ax.set_title("Root Mean Square Fluctuation per Residue", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(threshold, color="red", linestyle="--", alpha=0.5,
                   label=f"Threshold: {threshold:.2f} Å")
        ax.scatter(residues[high_fluct], rmsf_values[high_fluct],
                   color="red", s=30, zorder=5, label="High Fluctuation")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- CSV export ---
        csv_path = save_path.replace(".png", ".csv")
        pd.DataFrame({
            "Residue_Number":    residues,
            "RMSF_Angstrom":     np.round(rmsf_values, 4),
            "High_Fluctuation":  high_fluct.astype(int),
        }).to_csv(csv_path, index=False)

        return {
            "plot":        save_path,
            "csv":         csv_path,
            "mean":        round(mean_rmsf, 3),
            "std":         round(std_rmsf, 3),
            "max_residue": max_res,
            "unit":        "Å",
        }

    # ------------------------------------------------------------------ Rg
    def calculate_rg(self):
        rg_values, time_ns = [], []
        for ts in self.u.trajectory:
            rg_values.append(self.protein.radius_of_gyration())
            time_ns.append(ts.time / 1000.0)
        return np.array(time_ns), np.array(rg_values)

    def plot_rg(self, save_path="rg.png"):
        time_ns, rg_values = self.calculate_rg()
        mean_rg = float(np.mean(rg_values))
        std_rg = float(np.std(rg_values))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ns, rg_values, linewidth=1.5, color="#F18F01")
        ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Radius of Gyration (Å)", fontsize=12, fontweight="bold")
        ax.set_title("Protein Compactness over Time", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(mean_rg, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_rg:.2f} Å")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- CSV export ---
        csv_path = save_path.replace(".png", ".csv")
        pd.DataFrame({
            "Frame":       np.arange(len(time_ns)),
            "Time_ns":     np.round(time_ns, 4),
            "Rg_Angstrom": np.round(rg_values, 4),
        }).to_csv(csv_path, index=False)

        return {
            "plot": save_path,
            "csv":  csv_path,
            "mean": round(mean_rg, 3),
            "std":  round(std_rg, 3),
            "unit": "Å",
        }

    # ------------------------------------------------------------------ FEL
    def calculate_free_energy_landscape(self, rmsd_values=None, rg_values=None,
                                        bins=50, temperature=300):
        if rmsd_values is None:
            _, rmsd_values = self.calculate_rmsd()
        if rg_values is None:
            _, rg_values = self.calculate_rg()

        hist, xedges, yedges = np.histogram2d(rmsd_values, rg_values, bins=bins)
        hist = hist / np.sum(hist)
        hist[hist == 0] = np.min(hist[hist > 0]) * 0.01
        kB = 0.001987
        RT = kB * temperature
        free_energy = -RT * np.log(hist)
        free_energy -= np.min(free_energy)
        return free_energy, xedges, yedges

    def plot_free_energy_landscape(self, save_path="fel.png", temperature=300):
        _, rmsd_values = self.calculate_rmsd()
        _, rg_values = self.calculate_rg()
        fel, xedges, yedges = self.calculate_free_energy_landscape(
            rmsd_values, rg_values, temperature=temperature)

        fel_smooth = gaussian_filter(fel, sigma=1.0)
        min_idx = np.unravel_index(np.argmin(fel_smooth), fel_smooth.shape)
        min_rmsd = float(xedges[min_idx[0]])
        min_rg = float(yedges[min_idx[1]])

        distances_to_min = np.sqrt((rmsd_values - min_rmsd) ** 2 + (rg_values - min_rg) ** 2)
        min_frame_idx = int(np.argmin(distances_to_min))
        min_time_ns = float(self.u.trajectory[min_frame_idx].time / 1000.0)
        actual_rmsd = float(rmsd_values[min_frame_idx])
        actual_rg = float(rg_values[min_frame_idx])

        fig, ax = plt.subplots(figsize=(6, 5))
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        levels = np.linspace(0, np.percentile(fel_smooth, 95), 20)
        contour = ax.contourf(X, Y, fel_smooth.T, levels=levels, cmap="viridis")
        ax.contour(X, Y, fel_smooth.T, levels=levels, colors="white",
                   linewidths=0.5, alpha=0.3)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Free Energy (kcal/mol)", fontsize=12, fontweight="bold")
        ax.set_xlabel("RMSD (Å)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Radius of Gyration (Å)", fontsize=12, fontweight="bold")
        ax.set_title("Free Energy Landscape", fontsize=14, fontweight="bold")
        ax.plot(xedges[min_idx[0]], yedges[min_idx[1]], "r*", markersize=20,
                label="Global Minimum", markeredgecolor="white", markeredgewidth=1)
        ax.plot(actual_rmsd, actual_rg, "yo", markersize=12,
                label=f"Frame {min_frame_idx} ({min_time_ns:.2f} ns)",
                markeredgecolor="black", markeredgewidth=1.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- CSV export ---
        # Per-frame table: RMSD, Rg, and the free energy at each frame's bin
        csv_path = save_path.replace(".png", ".csv")
        rmsd_bin_idx = np.clip(np.digitize(rmsd_values, xedges[:-1]) - 1,
                               0, fel_smooth.shape[0] - 1)
        rg_bin_idx   = np.clip(np.digitize(rg_values,   yedges[:-1]) - 1,
                               0, fel_smooth.shape[1] - 1)
        frame_fel = fel_smooth[rmsd_bin_idx, rg_bin_idx]

        pd.DataFrame({
            "Frame":              np.arange(len(rmsd_values)),
            "RMSD_Angstrom":      np.round(rmsd_values, 4),
            "Rg_Angstrom":        np.round(rg_values,   4),
            "FreeEnergy_kcalmol": np.round(frame_fel,   4),
        }).to_csv(csv_path, index=False)

        return {
            "plot":        save_path,
            "csv":         csv_path,
            "min_frame":   min_frame_idx,
            "min_time_ns": round(min_time_ns, 3),
            "min_rmsd":    round(min_rmsd, 3),
            "min_rg":      round(min_rg, 3),
            "unit":        "kcal/mol",
        }

    # ------------------------------------------------------------------ Binding Energy
    def calculate_binding_energy_mm(self):
        if len(self.ligand) == 0:
            return None, None

        k_e = 138.935458
        epsilon_lj = 0.5
        sigma = 0.35

        energies, time_ns = [], []
        for ts in self.u.trajectory:
            distances_angstrom = mda.lib.distances.distance_array(
                self.protein.positions, self.ligand.positions)
            distances_nm = np.maximum(distances_angstrom / 10.0, 0.01)
            elec_energy = k_e * 0.5 * 0.5 / distances_nm
            sigma_r = sigma / distances_nm
            vdw_energy = 4 * epsilon_lj * (sigma_r ** 12 - sigma_r ** 6)
            energies.append(float(np.sum(elec_energy) + np.sum(vdw_energy)))
            time_ns.append(ts.time / 1000.0)

        return np.array(time_ns), np.array(energies)

    def plot_binding_energy(self, save_path="binding_energy.png"):
        time_ns, energies = self.calculate_binding_energy_mm()
        if energies is None:
            return None

        mean_e = float(np.mean(energies))
        std_e  = float(np.std(energies))

        # Scientific notation for the legend
        mean_sci = _to_sci(mean_e)
        std_sci  = _to_sci(std_e)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ns, energies, linewidth=1.5, color="#6A4C93")
        ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Interaction Energy (kJ/mol)", fontsize=12, fontweight="bold")
        ax.set_title("Protein-Ligand Interaction Energy", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(mean_e, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_sci} kJ/mol\nSD:    {std_sci} kJ/mol")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- CSV export ---
        csv_path = save_path.replace(".png", ".csv")
        pd.DataFrame({
            "Frame":                    np.arange(len(time_ns)),
            "Time_ns":                  np.round(time_ns, 4),
            "InteractionEnergy_kJmol":  np.round(energies, 4),
        }).to_csv(csv_path, index=False)

        return {
            "plot":     save_path,
            "csv":      csv_path,
            "mean":     round(mean_e, 3),
            "std":      round(std_e, 3),
            "mean_sci": mean_sci,
            "std_sci":  std_sci,
            "unit":     "kJ/mol",
        }

    # ------------------------------------------------------------------ P-L Distance
    def calculate_protein_ligand_distance(self, protein_sel="name CA", ligand_sel=None):
        if ligand_sel:
            ligand = self.u.select_atoms(ligand_sel)
        else:
            ligand = self.ligand

        if len(ligand) == 0:
            return None, None

        protein_atoms = self.protein.select_atoms(protein_sel)
        distances, time_ns = [], []
        for ts in self.u.trajectory:
            dist_array = mda.lib.distances.distance_array(
                protein_atoms.positions, ligand.positions)
            distances.append(float(np.min(dist_array)))
            time_ns.append(ts.time / 1000.0)

        return np.array(time_ns), np.array(distances)

    def plot_protein_ligand_distance(self, save_path="pl_distance.png"):
        time_ns, distances = self.calculate_protein_ligand_distance()
        if distances is None:
            return None

        mean_dist = float(np.mean(distances))
        contact_pct = float(np.mean(distances < 4.0) * 100)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_ns, distances, linewidth=1.5, color="#1D3557")
        ax.set_xlabel("Time (ns)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Minimum Distance (Å)", fontsize=12, fontweight="bold")
        ax.set_title("Protein-Ligand Minimum Distance", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(mean_dist, color="red", linestyle="--", alpha=0.7,
                   label=f"Mean: {mean_dist:.2f} Å")
        ax.axhline(4.0, color="green", linestyle=":", alpha=0.5,
                   label="Contact Threshold (4Å)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- CSV export ---
        csv_path = save_path.replace(".png", ".csv")
        pd.DataFrame({
            "Frame":                np.arange(len(time_ns)),
            "Time_ns":              np.round(time_ns, 4),
            "MinDistance_Angstrom": np.round(distances, 4),
        }).to_csv(csv_path, index=False)

        return {
            "plot":        save_path,
            "csv":         csv_path,
            "mean":        round(mean_dist, 3),
            "contact_pct": round(contact_pct, 1),
            "unit":        "Å",
        }


def extract_frame_from_trajectory(topology_file, trajectory_file, time_ns,
                                   output_pdb="extracted_frame.pdb",
                                   selection="all", dt_in_ps=None):
    if dt_in_ps:
        u = mda.Universe(topology_file, trajectory_file, dt=dt_in_ps)
    else:
        u = mda.Universe(topology_file, trajectory_file)
    times_ns = np.array([ts.time / 1000.0 for ts in u.trajectory])
    frame_idx = int(np.argmin(np.abs(times_ns - time_ns)))
    actual_time = float(times_ns[frame_idx])
    u.trajectory[frame_idx]
    atoms = u.select_atoms(selection)
    atoms.write(output_pdb)
    return frame_idx, actual_time
