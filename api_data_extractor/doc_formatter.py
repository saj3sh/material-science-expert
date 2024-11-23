from emmet.core.summary import SummaryDoc, Structure


def _format_structure(structure: Structure):
    lattice_params = structure.lattice.abc  # a, b, c
    lattice_angles = structure.lattice.angles  # alpha, beta, gamma
    num_sites = len(structure)
    species_summary = ", ".join(set(str(site.species)
                                for site in structure))
    # Can be modified if specific types are distinguishable
    structure_type = "3D Periodic Structure"

    return (
        f"{structure_type} with {num_sites} sites, "
        f"lattice parameters: a = {lattice_params[0]:.2f} Å, b = {lattice_params[1]:.2f} Å, and c = {lattice_params[2]:.2f} Å, "
        f"lattice angles: α={lattice_angles[0]: .2f}°, β={lattice_angles[1]: .2f}°, γ={lattice_angles[2]: .2f}°, "
        f"Species present: {species_summary}, "
        f"Charge: {structure.charge if structure.charge is not None else 'Neutral'}"
    )


def _format_decomposes_to(decomposes_to):
    if not decomposes_to:
        return ""

    decomposition_details = []
    for product in decomposes_to:
        parts = []
        if product.material_id:
            parts.append(product.material_id)
        if product.formula:
            parts.append(product.formula)
        if product.amount is not None:
            parts.append(
                f"(decomposition amount={product.amount: .3f} formula units)")
        formatted_parts = " ".join(filter(None, parts))
        decomposition_details.append(formatted_parts)

    return f"Decomposes to: {', '.join(decomposition_details)}"


def format_summary_doc(doc: SummaryDoc):
    description = []
    # Material ID and structure
    description.append(f"Material ID: {doc.material_id}")
    # theoritical?
    if doc.theoretical:
        description.append(f"The material is theoretical")
    if doc.structure:
        description.append(
            f"Structure: {_format_structure(doc.structure)}")
    # Energy properties
    energy_data = []
    if doc.uncorrected_energy_per_atom is not None:
        energy_data.append(
            f"uncorrected energy per atom = {doc.uncorrected_energy_per_atom:.3f} eV/atom")
    if doc.energy_per_atom is not None:
        energy_data.append(
            f"corrected energy per atom = {doc.energy_per_atom:.3f} eV/atom")
    if doc.formation_energy_per_atom is not None:
        energy_data.append(
            f"formation energy per atom = {doc.formation_energy_per_atom:.3f} eV/atom")
    if doc.energy_above_hull is not None:
        energy_data.append(
            f"energy above hull = {doc.energy_above_hull:.3f} eV/atom")
    if doc.equilibrium_reaction_energy_per_atom is not None:
        energy_data.append(
            f"equilibrium reaction energy = {doc.equilibrium_reaction_energy_per_atom:.3f} eV")
    if energy_data:
        description.append(
            f"Energy properties: {', '.join(energy_data)}")
    description.append(
        f"Stability: {'Stable' if doc.is_stable else 'Unstable'}")
    # Electronic properties
    electronic_data = []
    if doc.band_gap is not None:
        electronic_data.append(
            f"band gap = {doc.band_gap:.3f} eV ({'direct' if doc.is_gap_direct else 'indirect'})")
    if doc.is_metal is not None:
        electronic_data.append(
            f"material type = {'metallic' if doc.is_metal else 'non-metallic'}")
    if electronic_data:
        description.append(
            f"Electronic properties: {', '.join(electronic_data)}")
    # Magnetic properties
    magnetic_data = []
    if doc.total_magnetization is not None:
        magnetic_data.append(
            f"total magnetization = {doc.total_magnetization:.3f} μB")
    if doc.total_magnetization_normalized_vol is not None:
        magnetic_data.append(
            f"normalized magnetization by volume = {doc.total_magnetization_normalized_vol:.3f} μB/Å³")
    if doc.num_magnetic_sites is not None:
        magnetic_data.append(
            f"magnetic site count = {doc.num_magnetic_sites}")
    if magnetic_data:
        description.append(
            f"Magnetic properties: {', '.join(magnetic_data)}")
    # Mechanical properties
    mechanical_data = []
    if doc.bulk_modulus:
        mechanical_data.append(
            f"bulk modulus (VRH) = {doc.bulk_modulus['vrh']} GPa")
    if doc.shear_modulus:
        mechanical_data.append(
            f"shear modulus (VRH) = {doc.shear_modulus['vrh']} GPa")
    if doc.universal_anisotropy is not None:
        mechanical_data.append(
            f"universal anisotropy = {doc.universal_anisotropy:.3f}")
    if mechanical_data:
        description.append(
            f"Mechanical properties: {', '.join(mechanical_data)}")
    # Surface properties
    surface_data = []
    if doc.weighted_surface_energy_EV_PER_ANG2 is not None:
        surface_data.append(
            f"surface energy = {doc.weighted_surface_energy_EV_PER_ANG2:.3f} eV/Å²")
    if doc.weighted_work_function is not None:
        surface_data.append(
            f"work function = {doc.weighted_work_function:.3f} eV")
    if surface_data:
        description.append(
            f"Surface properties: {', '.join(surface_data)}")
    # decomposition info
    if doc.possible_species:
        description.append(
            f"Possible charged species: {', '.join(doc.possible_species)}")
    if doc.decomposes_to:
        description.append(_format_decomposes_to(doc.decomposes_to))
    # if self.database_IDs:
    #     description.append(f"Database IDs: {', '.join(self.database_IDs)}")

    return (doc.material_id, "; ".join(description).strip())


__all__ = [format_summary_doc]