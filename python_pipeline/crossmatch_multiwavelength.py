import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import math
import time
import os

def get_row_coords(row, colnames):
    """Helper to robustly extract RA and Dec from a table row in decimal degrees."""
    ra_keys = ['RA_ICRS', 'RAJ2000', 'ra', 'RAJ', '_RAJ2000']
    dec_keys = ['DE_ICRS', 'DEJ2000', 'dec', 'DEJ', '_DEJ2000']
    
    ra_col = next((k for k in ra_keys if k in colnames), None)
    dec_col = next((k for k in dec_keys if k in colnames), None)
    
    if ra_col is not None and dec_col is not None:
        val_ra = row[ra_col]
        val_dec = row[dec_col]
        # Handle masked values
        if hasattr(val_ra, 'mask') and val_ra.mask:
            raise ValueError("RA coordinate is masked")
        if hasattr(val_dec, 'mask') and val_dec.mask:
            raise ValueError("Dec coordinate is masked")
        return float(val_ra), float(val_dec)
    raise KeyError(f"Could not find coordinate columns in table. Available columns: {colnames}")

def clean_float(val):
    """Helper to cleanly convert a table cell to float, handling masked values."""
    if val is None or pd.isna(val) or (hasattr(val, 'mask') and val.mask) or str(val) == '--':
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def crossmatch_sources():
    print("=== Loading Sources for Multi-Wavelength Crossmatch ===")
    
    # 1. Load training eligible
    train_df = pd.read_csv("data/training_eligible.csv")
    train_df['Source_Name'] = train_df['Source_Name'].astype(str).str.strip()
    
    # 2. Load all 319 new DR3 sources
    dr2 = pd.read_csv("data/4LAC-DR2.csv")
    dr2_names = set(dr2['Source_Name'].astype(str).str.strip())
    
    from astropy.io import fits
    with fits.open("data/table-4LAC-DR3-h.fits") as h_hdul:
        dr3_h = pd.DataFrame(h_hdul[1].data)
    with fits.open("data/table-4LAC-DR3-l.fits") as l_hdul:
        dr3_l = pd.DataFrame(l_hdul[1].data)
    dr3 = pd.concat([dr3_h, dr3_l], ignore_index=True)
    for col in dr3.columns:
        if dr3[col].dtype == object:
            dr3[col] = dr3[col].astype(str).str.strip()
            
    new_names = set(dr3['Source_Name']) - dr2_names
    new_sources_df = dr3[dr3['Source_Name'].isin(new_names)].copy()
    
    # Merge Gaia magnitudes downloaded for new sources
    gaia_new = pd.read_csv("data/gaia_magnitudes_dr3_new.csv")
    gaia_new['Source_Name'] = gaia_new['Source_Name'].astype(str).str.strip()
    new_sources_df = pd.merge(new_sources_df, gaia_new, on="Source_Name", how="left")
    
    # Let's read existing matches if available to skip them
    existing_mw = None
    if os.path.exists("data/multi_wavelength_matches.csv"):
        existing_mw = pd.read_csv("data/multi_wavelength_matches.csv")
        existing_mw['Source_Name'] = existing_mw['Source_Name'].astype(str).str.strip()
        print(f"Loaded {len(existing_mw)} existing matches from data/multi_wavelength_matches.csv.")
        
    # Load generalization set
    gen_df = pd.read_csv("data/generalization_set.csv")
    gen_df['Source_Name'] = gen_df['Source_Name'].astype(str).str.strip()
    
    print(f"Loaded {len(train_df)} training sources, {len(new_sources_df)} new DR3 sources, and {len(gen_df)} generalization sources.")
    
    # Prepare coordinates list
    sources = []
    
    # Function to add source if not already matched
    def add_source(name, ra, dec, dataset):
        if existing_mw is not None and 'sdss_u' in existing_mw.columns and name in existing_mw['Source_Name'].values:
            return
        sources.append({
            'Source_Name': name,
            'ra': float(ra),
            'dec': float(dec),
            'dataset': dataset
        })
        
    for idx, row in train_df.iterrows():
        ra = row['RA_Counterpart'] if pd.notna(row['RA_Counterpart']) else row['RAJ2000']
        dec = row['DEC_Counterpart'] if pd.notna(row['DEC_Counterpart']) else row['DEJ2000']
        add_source(row['Source_Name'], ra, dec, 'train')
        
    for idx, row in new_sources_df.iterrows():
        ra = row['RA_Counterpart'] if pd.notna(row['RA_Counterpart']) else row['RAJ2000']
        dec = row['DEC_Counterpart'] if pd.notna(row['DEC_Counterpart']) else row['DEJ2000']
        add_source(row['Source_Name'], ra, dec, 'new_dr3')
        
    for idx, row in gen_df.iterrows():
        ra = row['RA_Counterpart'] if pd.notna(row['RA_Counterpart']) else row['RAJ2000']
        dec = row['DEC_Counterpart'] if pd.notna(row['DEC_Counterpart']) else row['DEJ2000']
        add_source(row['Source_Name'], ra, dec, 'generalization_dr2')
        
    sources_df = pd.DataFrame(sources)
    print(f"Total new coordinates to query: {len(sources_df)}")
    
    # Configure Vizier
    Vizier.ROW_LIMIT = 5
    Vizier.TIMEOUT = 60
    
    if len(sources_df) > 0:
        chunk_size = 100
        results = []
        
        catalogs = {
            'wise': 'II/328/allwise',
            'nvss': 'VIII/65/nvss',
            'swift': 'IX/58/2sxps',
            'sdss': 'V/154/sdss16',
            'ps1': 'II/349/ps1'
        }
        
        print("\n--- Starting Crossmatch Queries ---")
        start_time = time.time()
        
        for i in range(0, len(sources_df), chunk_size):
            chunk = sources_df.iloc[i:i+chunk_size]
            chunk_coords = SkyCoord(ra=chunk['ra'].values, dec=chunk['dec'].values, unit=(u.deg, u.deg))
            
            # Initialize dictionaries to hold matched values for this chunk
            wise_matches = {name: {'W1mag': np.nan, 'W2mag': np.nan, 'W3mag': np.nan, 'W4mag': np.nan} for name in chunk['Source_Name']}
            nvss_matches = {name: {'S1.4': np.nan} for name in chunk['Source_Name']}
            swift_matches = {name: {'CR0': np.nan} for name in chunk['Source_Name']}
            sdss_matches = {name: {'sdss_u': np.nan, 'sdss_g': np.nan, 'sdss_r': np.nan, 'sdss_i': np.nan, 'sdss_z': np.nan} for name in chunk['Source_Name']}
            ps1_matches = {name: {'ps1_g': np.nan, 'ps1_r': np.nan, 'ps1_i': np.nan, 'ps1_z': np.nan, 'ps1_y': np.nan} for name in chunk['Source_Name']}
            
            # 1. AllWISE query (5 arcseconds)
            try:
                res_wise = Vizier.query_region(chunk_coords, radius=5.0*u.arcsec, catalog=catalogs['wise'])
                if res_wise and len(res_wise) > 0:
                    tbl = res_wise[0]
                    for row in tbl:
                        match_idx = row['_q'] - 1
                        if match_idx < len(chunk):
                            src_name = chunk.iloc[match_idx]['Source_Name']
                            s_ra = chunk.iloc[match_idx]['ra']
                            s_dec = chunk.iloc[match_idx]['dec']
                            
                            try:
                                m_ra, m_dec = get_row_coords(row, tbl.colnames)
                                dist = math.sqrt((m_ra - s_ra)**2 + (m_dec - s_dec)**2)
                            except Exception:
                                dist = 999.0
                                
                            current = wise_matches[src_name]
                            if pd.isna(current['W1mag']) or dist < current.get('dist', float('inf')):
                                wise_matches[src_name] = {
                                    'W1mag': clean_float(row['W1mag']) if 'W1mag' in tbl.colnames else np.nan,
                                    'W2mag': clean_float(row['W2mag']) if 'W2mag' in tbl.colnames else np.nan,
                                    'W3mag': clean_float(row['W3mag']) if 'W3mag' in tbl.colnames else np.nan,
                                    'W4mag': clean_float(row['W4mag']) if 'W4mag' in tbl.colnames else np.nan,
                                    'dist': dist
                                }
            except Exception as e:
                print(f"Error on AllWISE chunk {i//chunk_size + 1}: {e}")
                
            # 2. NVSS query (10 arcseconds)
            try:
                res_nvss = Vizier.query_region(chunk_coords, radius=10.0*u.arcsec, catalog=catalogs['nvss'])
                if res_nvss and len(res_nvss) > 0:
                    tbl = res_nvss[0]
                    for row in tbl:
                        match_idx = row['_q'] - 1
                        if match_idx < len(chunk):
                            src_name = chunk.iloc[match_idx]['Source_Name']
                            s_ra = chunk.iloc[match_idx]['ra']
                            s_dec = chunk.iloc[match_idx]['dec']
                            
                            try:
                                ra_col = 'RAJ2000' if 'RAJ2000' in tbl.colnames else ('RA_ICRS' if 'RA_ICRS' in tbl.colnames else None)
                                dec_col = 'DEJ2000' if 'DEJ2000' in tbl.colnames else ('DE_ICRS' if 'DE_ICRS' in tbl.colnames else None)
                                c_match = SkyCoord(str(row[ra_col]), str(row[dec_col]), unit=(u.hourangle, u.deg))
                                match_ra = c_match.ra.deg
                                match_dec = c_match.dec.deg
                                dist = math.sqrt((match_ra - s_ra)**2 + (match_dec - s_dec)**2)
                            except Exception:
                                dist = 999.0
                                
                            current = nvss_matches[src_name]
                            if pd.isna(current['S1.4']) or dist < current.get('dist', float('inf')):
                                nvss_matches[src_name] = {
                                    'S1.4': clean_float(row['S1.4']) if 'S1.4' in tbl.colnames else np.nan,
                                    'dist': dist
                                }
            except Exception as e:
                print(f"Error on NVSS chunk {i//chunk_size + 1}: {e}")
                
            # 3. Swift-XRT query (10 arcseconds)
            try:
                res_swift = Vizier.query_region(chunk_coords, radius=10.0*u.arcsec, catalog=catalogs['swift'])
                if res_swift and len(res_swift) > 0:
                    tbl = res_swift[0]
                    for row in tbl:
                        match_idx = row['_q'] - 1
                        if match_idx < len(chunk):
                            src_name = chunk.iloc[match_idx]['Source_Name']
                            s_ra = chunk.iloc[match_idx]['ra']
                            s_dec = chunk.iloc[match_idx]['dec']
                            
                            try:
                                m_ra, m_dec = get_row_coords(row, tbl.colnames)
                                dist = math.sqrt((m_ra - s_ra)**2 + (m_dec - s_dec)**2)
                            except Exception:
                                dist = 999.0
                                
                            current = swift_matches[src_name]
                            if pd.isna(current['CR0']) or dist < current.get('dist', float('inf')):
                                swift_matches[src_name] = {
                                    'CR0': clean_float(row['CR0']) if 'CR0' in tbl.colnames else np.nan,
                                    'dist': dist
                                }
            except Exception as e:
                print(f"Error on Swift-XRT chunk {i//chunk_size + 1}: {e}")
                
            # 4. SDSS DR16 query (5 arcseconds)
            try:
                res_sdss = Vizier.query_region(chunk_coords, radius=5.0*u.arcsec, catalog=catalogs['sdss'])
                if res_sdss and len(res_sdss) > 0:
                    tbl = res_sdss[0]
                    for row in tbl:
                        match_idx = row['_q'] - 1
                        if match_idx < len(chunk):
                            src_name = chunk.iloc[match_idx]['Source_Name']
                            s_ra = chunk.iloc[match_idx]['ra']
                            s_dec = chunk.iloc[match_idx]['dec']
                            
                            try:
                                m_ra, m_dec = get_row_coords(row, tbl.colnames)
                                dist = math.sqrt((m_ra - s_ra)**2 + (m_dec - s_dec)**2)
                            except Exception:
                                dist = 999.0
                                
                            current = sdss_matches[src_name]
                            if pd.isna(current['sdss_g']) or dist < current.get('dist', float('inf')):
                                sdss_matches[src_name] = {
                                    'sdss_u': clean_float(row['umag']) if 'umag' in tbl.colnames else np.nan,
                                    'sdss_g': clean_float(row['gmag']) if 'gmag' in tbl.colnames else np.nan,
                                    'sdss_r': clean_float(row['rmag']) if 'rmag' in tbl.colnames else np.nan,
                                    'sdss_i': clean_float(row['imag']) if 'imag' in tbl.colnames else np.nan,
                                    'sdss_z': clean_float(row['zmag']) if 'zmag' in tbl.colnames else np.nan,
                                    'dist': dist
                                }
            except Exception as e:
                print(f"Error on SDSS chunk {i//chunk_size + 1}: {e}")
                
            # 5. Pan-STARRS1 query (5 arcseconds)
            try:
                res_ps1 = Vizier.query_region(chunk_coords, radius=5.0*u.arcsec, catalog=catalogs['ps1'])
                if res_ps1 and len(res_ps1) > 0:
                    tbl = res_ps1[0]
                    for row in tbl:
                        match_idx = row['_q'] - 1
                        if match_idx < len(chunk):
                            src_name = chunk.iloc[match_idx]['Source_Name']
                            s_ra = chunk.iloc[match_idx]['ra']
                            s_dec = chunk.iloc[match_idx]['dec']
                            
                            try:
                                m_ra, m_dec = get_row_coords(row, tbl.colnames)
                                dist = math.sqrt((m_ra - s_ra)**2 + (m_dec - s_dec)**2)
                            except Exception:
                                dist = 999.0
                                
                            current = ps1_matches[src_name]
                            if pd.isna(current['ps1_g']) or dist < current.get('dist', float('inf')):
                                ps1_matches[src_name] = {
                                    'ps1_g': clean_float(row['gmag']) if 'gmag' in tbl.colnames else np.nan,
                                    'ps1_r': clean_float(row['rmag']) if 'rmag' in tbl.colnames else np.nan,
                                    'ps1_i': clean_float(row['imag']) if 'imag' in tbl.colnames else np.nan,
                                    'ps1_z': clean_float(row['zmag']) if 'zmag' in tbl.colnames else np.nan,
                                    'ps1_y': clean_float(row['ymag']) if 'ymag' in tbl.colnames else np.nan,
                                    'dist': dist
                                }
            except Exception as e:
                print(f"Error on Pan-STARRS1 chunk {i//chunk_size + 1}: {e}")
                
            # Collect results for this chunk
            for src_name in chunk['Source_Name']:
                w = wise_matches[src_name]
                n = nvss_matches[src_name]
                s = swift_matches[src_name]
                sd = sdss_matches[src_name]
                p = ps1_matches[src_name]
                results.append({
                    'Source_Name': src_name,
                    'W1mag': w.get('W1mag', np.nan),
                    'W2mag': w.get('W2mag', np.nan),
                    'W3mag': w.get('W3mag', np.nan),
                    'W4mag': w.get('W4mag', np.nan),
                    'S1.4': n.get('S1.4', np.nan),
                    'CR0': s.get('CR0', np.nan),
                    'sdss_u': sd.get('sdss_u', np.nan),
                    'sdss_g': sd.get('sdss_g', np.nan),
                    'sdss_r': sd.get('sdss_r', np.nan),
                    'sdss_i': sd.get('sdss_i', np.nan),
                    'sdss_z': sd.get('sdss_z', np.nan),
                    'ps1_g': p.get('ps1_g', np.nan),
                    'ps1_r': p.get('ps1_r', np.nan),
                    'ps1_i': p.get('ps1_i', np.nan),
                    'ps1_z': p.get('ps1_z', np.nan),
                    'ps1_y': p.get('ps1_y', np.nan)
                })
                
            print(f"Processed chunk {i//chunk_size + 1} / {math.ceil(len(sources_df)/chunk_size)}")
            time.sleep(0.5) # rate limit friendly
            
        new_mw_df = pd.DataFrame(results)
        if existing_mw is not None:
            out_df = pd.concat([existing_mw, new_mw_df], ignore_index=True)
        else:
            out_df = new_mw_df
        out_df = out_df.drop_duplicates(subset=['Source_Name'], keep='last')
        out_df.to_csv("data/multi_wavelength_matches.csv", index=False)
        
        elapsed = time.time() - start_time
        print(f"\nSaved {len(out_df)} matched records to data/multi_wavelength_matches.csv in {elapsed:.1f} seconds.")
    else:
        print("\nAll sources are already crossmatched. No queries performed.")
        out_df = existing_mw
        
    # Print match rates
    print("Match rates:")
    print(f"  AllWISE (any band): {out_df['W1mag'].notna().sum()} / {len(out_df)} ({out_df['W1mag'].notna().sum()/len(out_df)*100:.1f}%)")
    print(f"  NVSS Radio Flux:     {out_df['S1.4'].notna().sum()} / {len(out_df)} ({out_df['S1.4'].notna().sum()/len(out_df)*100:.1f}%)")
    print(f"  Swift X-ray CR:      {out_df['CR0'].notna().sum()} / {len(out_df)} ({out_df['CR0'].notna().sum()/len(out_df)*100:.1f}%)")
    print(f"  SDSS (u band):       {out_df['sdss_u'].notna().sum()} / {len(out_df)} ({out_df['sdss_u'].notna().sum()/len(out_df)*100:.1f}%)")
    print(f"  Pan-STARRS1 (g band): {out_df['ps1_g'].notna().sum()} / {len(out_df)} ({out_df['ps1_g'].notna().sum()/len(out_df)*100:.1f}%)")

if __name__ == "__main__":
    crossmatch_sources()
