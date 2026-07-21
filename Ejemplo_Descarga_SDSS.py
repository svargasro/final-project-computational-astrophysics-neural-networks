import numpy as np
from scipy.interpolate import interp1d
import warnings

# Se necesita instalar astroquery y astropy si no están instalados:
# pip install astroquery astropy
try:
    from astroquery.sdss import SDSS
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    
except ImportError:
    print("Por favor, instala astroquery y astropy corriendo: pip install astroquery astropy")

def descargar_y_procesar_sdss(n_espectros=100, estratificado=False):
    """
    Descarga espectros reales de galaxias de SDSS y los interpola a una 
    cuadrícula fija de longitudes de onda, listos para una Red Neuronal.
    """
    print(f"Haciendo consulta a SDSS para obtener {n_espectros} galaxias...")
    
    # Sistema de reintentos para evitar errores 503 (Servidor Ocupado)
    import time
    from astropy.table import vstack
    
    max_reintentos = 5
    resultados_totales = None
    
    if estratificado:
        num_bloques = 5
        espectros_por_bloque = n_espectros // num_bloques
        limites_z = np.linspace(0.01, 1.0, num_bloques + 1)
        
        print(f"Modo Estratificado activado: {num_bloques} bloques de {espectros_por_bloque} espectros.")
        tablas = []
        
        for i in range(num_bloques):
            z_min = limites_z[i]
            z_max = limites_z[i+1]
            print(f"Consultando bloque {i+1}/{num_bloques}: z entre {z_min:.2f} y {z_max:.2f}")
            
            query = f"""
                SELECT TOP {espectros_por_bloque}
                    specObjID, ra, dec, z, plate, mjd, fiberID, run2d
                FROM SpecObj
                WHERE class = 'GALAXY' AND zWarning = 0 AND z BETWEEN {z_min} AND {z_max}
            """
            
            resultados_bloque = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for intento in range(max_reintentos):
                    try:
                        resultados_bloque = SDSS.query_sql(query)
                        break
                    except Exception as e:
                        print(f"Error de conexión (Intento {intento+1}/{max_reintentos}): {e}")
                        if intento < max_reintentos - 1:
                            time.sleep(10)
                        else:
                            print("Servidor ocupado. Abortando.")
                            return None, None
                            
            if resultados_bloque is not None:
                tablas.append(resultados_bloque)
                
        if len(tablas) > 0:
            resultados_totales = vstack(tablas)
            
    else:
        # 1. Hacemos una consulta SQL a SDSS normal
        query = f"""
            SELECT TOP {n_espectros}
                specObjID, ra, dec, z, plate, mjd, fiberID, run2d
            FROM SpecObj
            WHERE class = 'GALAXY' AND zWarning = 0 AND z BETWEEN 0.01 AND 1.0
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for intento in range(max_reintentos):
                try:
                    resultados_totales = SDSS.query_sql(query)
                    break # Si funciona, salimos del bucle
                except Exception as e:
                    print(f"Error de conexión con SDSS (Intento {intento+1}/{max_reintentos}): {e}")
                    if intento < max_reintentos - 1:
                        print("Esperando 10 segundos antes de reintentar...")
                        time.sleep(10)
                    else:
                        print("El servidor de SDSS está caído o bloqueando la conexión. Intenta más tarde.")
                        return None, None
                        
    if resultados_totales is None or len(resultados_totales) == 0:
        return None, None
        
    print(f"Descargando {len(resultados_totales)} espectros (archivos FITS)... esto puede tomar un momento.")
    
    # 2. Descargamos los archivos FITS de los espectros
    # SDSS.get_spectra descarga los archivos localmente (los guarda en caché)
    espectros_fits = SDSS.get_spectra(matches=resultados_totales)
    
    # 3. Preparamos nuestro dominio (el mismo de tu Red Neuronal)
    # Tu red usa de 400 nm a 900 nm en 1000 pasos. 
    # SDSS usa Angstroms (1 nm = 10 A), así que de 4000 A a 9000 A.
    lambda_min_nm = 400
    lambda_max_nm = 900
    N_lambda = 1000
    lambda_instrument_nm = np.linspace(lambda_min_nm, lambda_max_nm, N_lambda)
    lambda_instrument_A = lambda_instrument_nm * 10 # Convertido a Angstroms para coincidir con SDSS
    
    # Matrices para guardar los datos listos para ML
    X_real = np.zeros((len(espectros_fits), N_lambda))
    y_real = np.zeros(len(espectros_fits))
    
    print("Procesando e interpolando los espectros FITS...")
    
    # 4. Leer cada FITS, extraer el flujo e interpolar
    for i, sp in enumerate(espectros_fits):
        # El archivo FITS de SDSS guarda los datos en la extensión 1
        datos = sp[1].data
        
        # Flujo (intensidad) y longitud de onda (viene en Log10(Angstroms))
        flujo = datos['flux']
        log_lam = datos['loglam']
        lam_A = 10**log_lam # Convertimos a Angstroms lineales
        
        # El redshift de este espectro (que sacamos de la tabla de resultados)
        z = resultados_totales['z'][i]
        
        # 5. Interpolación
        # Como los arrays lam_A tienen distintos tamaños y puntos exactos,
        # creamos una función matemática continua con los datos reales...
        interpolador = interp1d(lam_A, flujo, kind='linear', bounds_error=False, fill_value=0.0)
        
        # ... y la evaluamos exactamente en nuestros 1000 puntos (de 4000 a 9000 A)
        flujo_interpolado = interpolador(lambda_instrument_A)
        
        # Guardamos en nuestras matrices
        X_real[i, :] = flujo_interpolado
        y_real[i] = z
        
    print("¡Procesamiento completo!")
    print(f"X_real shape: {X_real.shape} (Flujo)")
    print(f"y_real shape: {y_real.shape} (Redshift)")
    
    return X_real, y_real

if __name__ == "__main__":
    import os
    from tensorflow.keras.models import load_model
    
        # 1. Configuración de carpetas y archivos locales en Colab
    CARPETA_DATOS = "datos_espectros"
    ARCHIVO_DATOS = os.path.join(CARPETA_DATOS, "espectros_sdss.npz")

    # Crear la carpeta local si no existe
    os.makedirs(CARPETA_DATOS, exist_ok=True)

    # 2. Comprobar si el archivo ya fue descargado y guardado
    if os.path.exists(ARCHIVO_DATOS):
        print(f"El archivo local ya existe en '{ARCHIVO_DATOS}'. Cargando datos sin descargar de nuevo...")
        datos = np.load(ARCHIVO_DATOS)
        X = datos['X']
        y_true = datos['y']
        print(f"Datos cargados exitosamente. Formato X: {X.shape}, y: {y_true.shape}")
    else:
        print("No se encontró el archivo local. Iniciando descarga y procesamiento de SDSS...")
        n_test = 100
        X, y_true = descargar_y_procesar_sdss(n_espectros=n_test)
        
        # Guardar en el disco local de Colab para futuras ejecuciones en esta misma sesión
        if X is not None:
            np.savez(ARCHIVO_DATOS, X=X, y=y_true)
            print(f"Espectros procesados y guardados localmente en '{ARCHIVO_DATOS}'.")

    
    
    if X is not None:
        # 2. Intentamos cargar tu modelo entrenado
        model_path = 'Estándar_NN.h5'
        if os.path.exists(model_path):
            print(f"\nCargando modelo desde {model_path}...")
            model = load_model(model_path)
            
            # 3. Preprocesamiento (Normalización de espectros)
            # Usamos la misma lógica que en tu notebook
            X_norm = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
            X_reshaped = X_norm[..., np.newaxis] # (N, 1000, 1)
            
            # 4. Predicciones
            print("Realizando predicciones...")
            y_pred_norm = model.predict(X_reshaped)
            
            # Nota: Como no tenemos el y_scaler guardado aquí, 
            # las predicciones estarán en escala normalizada.
            # Pero podemos ver la correlación o imprimir algunos valores.
            print("\nPrimeras 10 comparaciones (z_real vs z_pred_normalizado):")
            print("Z_REAL | Z_PRED_NORM")
            print("-" * 25)
            for i in range(min(10, len(y_true))):
                print(f"{y_true[i]:.4f} | {y_pred_norm[i][0]:.4f}")
                
            # Calculamos una correlación simple para ver si "sigue" la tendencia
            from scipy.stats import pearsonr
            corr, _ = pearsonr(y_true, y_pred_norm.flatten())
            print(f"\nCorrelación de Pearson entre z_real y z_pred: {corr:.4f}")
        else:
            print(f"\nNo se encontró el archivo {model_path}. Entrena el modelo en el notebook primero.")



