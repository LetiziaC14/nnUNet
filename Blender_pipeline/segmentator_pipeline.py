# coding: utf-8
# segmentator_pipeline.py
import sys
import os

try:
    import Blender_pipeline.config as config
    print("DEBUG: config importato.")
    import segmentator_ops
    print("DEBUG: segmentator_ops importato.")
    import Blender_pipeline.utils as utils
    print("DEBUG: utils importato.")
except ImportError as e:
    print(f"ERRORE CRITICO: Impossibile importare i moduli necessari. Assicurati che tutte le dipendenze siano installate (pip install -r requirements.txt). Dettagli: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERRORE INATTESO durante l'importazione dei moduli: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def execute_segmentator_pipeline():
    print("DEBUG: segmentator_pipeline.py in esecuzione .")
    try:
        """
        Orchestra l'intera pipeline di segmentazione, dalla lettura della segmentazione
        all'esportazione dei file STL e del manifest.
        """
        print(f"--- Avvio Pipeline di Segmentazione per: {config.CLIENT_ID}, CASO: {config.PROJECT_SESSION_ID} ---")

        # 1. Trova e Prepara i File di Input NIfTI
        segmented_nii_path = "C:\\Users\\letiz\\Documents\\Huvant\\inference_new_case_fold0\\fullsize\\99997.nii.gz"  #to be replaced with actual path finding logic
    
        if segmented_nii_path:
            print(f"\nDEBUG: File NIfTI segmentato disponibile in: {segmented_nii_path}")
            print(f"DEBUG: Caricamento della Class Map")
            segment_id_to_name_map = {
                0: "Background",
                1: "Kidney",
                2: "Tumor",
                3: "Cyst"
            }
            print(f"DEBUG: Class Map (Segment ID to Name Table) caricata: {len(segment_id_to_name_map)} entries.")


            # 1.5 Determina quali segmenti hanno un volume effettivo nel NIfTI
            valid_segment_ids = segmentator_ops.get_present_segment_ids(
                segmented_nii_path,
                segment_id_to_name_map # Passiamo questa mappa per i nomi nel debug
            )
            print(f"DEBUG: Segmenti con volume effettivo presenti: {sorted(list(valid_segment_ids))}")

            # 2. Inizializza la struttura dati centrale per tutti i segmenti *presenti*
            print("\nDEBUG: Inizializzazione struttura 'all_segment_data'")
            SNOMED_LOOKUP = {
            "kidney_parenchyma": {
                "category": "Anatomical structure",
                "type": "Kidney",
                "type_modifier": None,
                "region": "Kidney",
                "type_code": "64033007",   # example SNOMED code
            },
            "kidney_tumor": {
                "category": "Pathological structure",
                "type": "Neoplasm",
                "type_modifier": None,
                "region": "Kidney",
                "type_code": "108369006",
            },
            "kidney_other_mass": {
                "category": "Pathological structure",
                "type": "Cyst",
                "type_modifier": None,
                "region": "Kidney",
                "type_code": "312850006",
            },
            }

            all_segment_data = {}
            for seg_id, seg_name in segment_id_to_name_map.items():
                if seg_id in valid_segment_ids:
                    # Take SNOMED details if available, otherwise default to Nones
                    snomed_details = SNOMED_LOOKUP.get(seg_name, {
                        "category": None,
                        "type": None,
                        "type_modifier": None,
                        "region": None,
                        "type_code": None,
                    })

                    all_segment_data[seg_name] = {
                        "id": seg_id,
                        "snomed_details": snomed_details,
                        "custom_parameters": {
                            "display_name": None,
                            "export_as_individual_mesh": None,
                            "biological_category": None,
                            "shader_ref": None,
                            "blend_file": None,
                            "blend_material": None,
                            "color_override": None
                        }
                    }

            print("DEBUG: Struttura 'all_segment_data' inizializzata")
           
            # 5. Carica le mappature dal file YAML (per le regole di export/combinazione)
            print(f"\nDEBUG: Caricamento dei dati SNOMED\n")
            segment_mappings_yaml = utils.read_yaml(config.SEGMENT_MAPPINGS_FILE)
            if not segment_mappings_yaml:
                print("Nessuna mappatura caricata o file non trovato.")
            else:
                print(f"DEBUG: YAML Mappings caricato da {config.SEGMENT_MAPPINGS_FILE}: {len(segment_mappings_yaml)} entries.\n")

            # --- Fase di Popolamento dei Custom Parameters per l'Export STL ---
            # Carica la 
            print("\nDEBUG:--- Fase: Popolamento dei Custom Parameters per l'export individuale / combinato ---")
            individual_mesh_rules = segment_mappings_yaml.get('individual_mesh_export', {})
            combined_mesh_rules = segment_mappings_yaml.get('combined_mesh_export', {})
            
            segmentator_ops.populate_custom_details_for_segments(all_segment_data, individual_mesh_rules, combined_mesh_rules)
            print("\nDEBUG:--- Popolamento Custom Parameters per l'export completato. ---")

            # --- Fase di Esportazione STL ---
            print(f"\n--- Esportazione STL ---")
            segmentator_ops.export_stl_from_multilabel_nii(
                nii_filepath=segmented_nii_path,
                all_segment_data=all_segment_data,
                combined_mesh_rules=combined_mesh_rules,
                output_dir=config.INPUT_MESH_DIR
            )

            # --- Fase di Scrittura del Manifest ---
            print(f"\nDEBUG:--- Scrittura del Manifest dei Segmenti ---")
            # Assicurati che la directory di output esista prima di scrivere il file
            output_dir = os.path.dirname(config.SEGMENTS_DATA_MANIFEST_FILE)
            os.makedirs(output_dir, exist_ok=True)
            
            utils.write_json(all_segment_data, config.SEGMENTS_DATA_MANIFEST_FILE)
            print(f"Manifest salvato in: {config.SEGMENTS_DATA_MANIFEST_FILE}")
            
            print("\n--- Pipeline di Segmentazione e Creazione Mesh STL completata con successo. ---")

        else: # segmented_nii_path e' None
            print("AVVISO: La segmentazione NIfTI non ha prodotto un file valido. La pipeline di esportazione STL verra' saltata.")

    except Exception as e:
        print(f"ERRORE CRITICO durante la pipeline di segmentazione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    execute_segmentator_pipeline()