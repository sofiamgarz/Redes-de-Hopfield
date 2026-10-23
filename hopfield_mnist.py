#!/usr/bin/env python3
"""Hopfield clásico para 100 imágenes MNIST; experimento reproducible.

Instalar: python -m pip install numpy==2.3.5 matplotlib==3.11.2
Ejecutar: python hopfield_mnist.py
La primera ejecución descarga MNIST. Después reutiliza mnist_100.npz;
--regenerar descarga MNIST y vuelve a seleccionar con semillas 42 y 43.
Los pesos de Hebb se calculan exactamente como J/N. La dinámica trabaja
con J entero: multiplicar todos los campos por N no cambia su signo.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import urllib.request
import numpy as np

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
N = 784
NIVELES = [0.0, 0.05, 0.10, 0.20]


def preparar_datos(destino, fuente, regenerar=False):
    """Selecciona 10 ejemplos/clase y divide 7/3 sin solapamientos."""
    if destino.exists() and not regenerar:
        with np.load(destino, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    if not fuente.exists():
        fuente.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, fuente)
    if hashlib.sha256(fuente.read_bytes()).hexdigest() != SHA256:
        raise ValueError('La suma SHA256 del archivo MNIST es incorrecta.')
    with np.load(fuente, allow_pickle=False) as f:
        imagenes, etiquetas = f['x_train'], f['y_train']
    seleccion = np.random.default_rng(42)
    division = np.random.default_rng(43)
    ids, entrenamiento = [], []
    for digito in range(10):
        candidatos = np.flatnonzero(etiquetas == digito)
        elegidos = seleccion.choice(candidatos, size=10, replace=False)
        orden = division.permutation(10)
        ids.extend(elegidos[orden])
        entrenamiento.extend([True] * 7 + [False] * 3)
    ids = np.array(ids, dtype=np.int32)
    datos = dict(images=imagenes[ids], labels=etiquetas[ids],
                 source_indices=ids, train_mask=np.array(entrenamiento))
    np.savez_compressed(destino, **datos)
    return datos


def binarizar(imagenes):
    """Normaliza a [0,1], umbraliza en 0.5 y aplana por filas a {-1,+1}."""
    return np.where(imagenes.reshape(-1, N) / 255.0 >= 0.5,
                    1, -1).astype(np.int32)


def entrenar(patrones):
    """Devuelve J = X.T @ X con diagonal cero; los pesos son W = J/N."""
    x = np.asarray(patrones, dtype=np.int32)  # Evita overflow de int8.
    j = x.T @ x
    np.fill_diagonal(j, 0)
    assert np.array_equal(j, j.T)
    assert not np.any(np.diag(j))
    return j


def energia(estado, j):
    """E(s) = -s.T @ W @ s / 2, con umbrales nulos y W = J/N."""
    s = np.asarray(estado, dtype=np.int64)
    return -float(s @ (j @ s)) / (2 * len(s))


def recuperar(inicial, j, max_barridos=100, traza=False):
    """Actualiza 1 neurona cada vez en orden fijo y conserva s_i si h_i=0.

    Se detiene después de un barrido completo sin cambios. Los campos se
    mantienen incrementalmente con enteros exactos. Se comprueba el descenso
    de energía en CADA cambio, además de verificarla al final de cada barrido.
    """
    s = np.asarray(inicial, dtype=np.int32).copy()
    h = j @ s
    e2 = -int(s.astype(np.int64) @ h)  # 2*N*E, entero exacto.
    energias = [e2 / (2 * len(s))]
    estados = [s.copy()]
    celda = [energias[0]] if traza else None
    cambios = []
    for barrido in range(1, max_barridos + 1):
        n_cambios = 0
        for i in range(len(s)):
            nuevo = 1 if h[i] > 0 else (-1 if h[i] < 0 else int(s[i]))
            delta = nuevo - int(s[i])
            if delta:
                variacion = -2 * delta * int(h[i])
                assert variacion < 0, 'La energía debe bajar al cambiar un bit.'
                e2 += variacion
                s[i] = nuevo
                h += delta * j[:, i]
                n_cambios += 1
            if traza:
                celda.append(e2 / (2 * len(s)))
        assert np.array_equal(h, j @ s)
        assert np.isclose(e2 / (2 * len(s)), energia(s, j), atol=1e-9)
        energias.append(e2 / (2 * len(s)))
        estados.append(s.copy())
        cambios.append(n_cambios)
        if n_cambios == 0:
            return dict(final=s, energies=energias, states=estados,
                        changes=cambios, sweeps=barrido,
                        converged=True, cell_energies=celda)
    return dict(final=s, energies=energias, states=estados,
                changes=cambios, sweeps=max_barridos,
                converged=False, cell_energies=celda)


def vecino(estado, patrones, etiquetas):
    """Etiqueta del vecino de Hamming; en empate, menor índice de training."""
    distancias = np.count_nonzero(patrones != estado, axis=1)
    indice = int(np.argmin(distancias))
    return int(etiquetas[indice]), indice, int(distancias[indice])


def matriz_confusion(reales, predichos):
    """Filas: clase real; columnas: predicción; orden de etiquetas 0 a 9."""
    matriz = np.zeros((10, 10), dtype=int)
    np.add.at(matriz, (reales, predichos), 1)
    return matriz


def comprobar_algoritmo():
    """Pruebas pequeñas: memoria única, energía, empates y paso asíncrono."""
    patron = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int32)
    j = entrenar(patron[None, :])
    inicial = patron.copy()
    inicial[2] *= -1
    resultado = recuperar(inicial, j, traza=True)
    assert np.array_equal(resultado['final'], patron)
    assert np.all(np.diff(resultado['cell_energies']) <= 0)
    assert np.array_equal(recuperar(patron, np.zeros((8, 8), dtype=int))['final'], patron)
    # Referencia independiente: recalcula el campo por producto escalar.
    rng = np.random.default_rng(123)
    patrones = rng.choice([-1, 1], size=(3, 12)).astype(np.int32)
    j = entrenar(patrones)
    inicial = rng.choice([-1, 1], size=12).astype(np.int32)
    s = inicial.copy()
    for _ in range(100):
        anterior = s.copy()
        for i in range(len(s)):
            campo = int(j[i] @ s)
            if campo:
                s[i] = 1 if campo > 0 else -1
        if np.array_equal(s, anterior):
            break
    assert np.array_equal(s, recuperar(inicial, j)['final'])


def ejecutar(datos, repeticiones=20):
    """Evalúa 30 tests por nivel y repetición; el entrenamiento queda fijo."""
    assert datos['images'].shape == (100, 28, 28)
    assert len(np.unique(datos['source_indices'])) == 100
    x, y, mask = binarizar(datos['images']), datos['labels'], datos['train_mask']
    assert np.array_equal(np.bincount(y), [10] * 10)
    assert np.array_equal(np.bincount(y[mask]), [7] * 10)
    assert np.array_equal(np.bincount(y[~mask]), [3] * 10)
    train, test, yt, ytest = x[mask], x[~mask], y[mask], y[~mask]
    j = entrenar(train)
    rows, resumen, ejemplos, finals = [], [], {}, []
    for nivel in NIVELES:
        rep_count = 1 if nivel == 0 else repeticiones
        aciertos, aciertos_base, cms, barridos = [], [], [], []
        for rep in range(rep_count):
            pred, base = [], []
            for i, original in enumerate(test):
                # Mismas permutaciones para niveles distintos: ruido anidado.
                rng = np.random.default_rng(np.random.SeedSequence([44, rep, i]))
                k = int(round(nivel * N))
                inicial = original.copy()
                inicial[rng.permutation(N)[:k]] *= -1
                r = recuperar(inicial, j, traza=(rep == 0 and i == 0))
                if not r['converged']:
                    raise RuntimeError('Límite alcanzado sin convergencia.')
                etiqueta, indice, dist = vecino(r['final'], train, yt)
                etiqueta_base, _, _ = vecino(inicial, train, yt)
                pred.append(etiqueta)
                base.append(etiqueta_base)
                barridos.append(r['sweeps'])
                finals.append(r['final'])
                rows.append(dict(noise=nivel, repetition=rep, test_index=i,
                    source_index=int(datos['source_indices'][~mask][i]),
                    true_label=int(ytest[i]), prediction=etiqueta,
                    baseline_prediction=etiqueta_base, flipped_bits=k,
                    sweeps=r['sweeps'], energy_initial=r['energies'][0],
                    energy_final=r['energies'][-1], nearest_train_index=indice,
                    nearest_hamming=dist))
                if rep == 0:
                    ejemplos[(nivel, i)] = dict(**r, initial=inicial,
                        original=original, nearest=train[indice], true=int(ytest[i]),
                        predicted=etiqueta)
            aciertos.append(float(np.mean(np.array(pred) == ytest)))
            aciertos_base.append(float(np.mean(np.array(base) == ytest)))
            cms.append(matriz_confusion(ytest, pred))
        resumen.append(dict(noise=nivel, flipped_bits=int(round(nivel * N)),
            effective_noise=round(nivel * N) / N, repetitions=rep_count,
            evaluations=30 * rep_count, accuracy_mean=float(np.mean(aciertos)),
            accuracy_std=float(np.std(aciertos, ddof=1)) if rep_count > 1 else 0,
            accuracy_first=aciertos[0], baseline_mean=float(np.mean(aciertos_base)),
            baseline_std=float(np.std(aciertos_base, ddof=1)) if rep_count > 1 else 0,
            baseline_first=aciertos_base[0], sweeps_mean=float(np.mean(barridos)),
            sweeps_min=int(min(barridos)), sweeps_max=int(max(barridos)),
            confusion_first=cms[0].tolist(),
            confusion_aggregate=np.sum(cms, axis=0).tolist()))
    overlap = (train @ train.T) / N
    pares = overlap[np.triu_indices(len(train), k=1)]
    estables = np.all(train * (train @ j.T) >= 0, axis=1)
    diagnostico = dict(train_background_fraction=float(np.mean(train == -1)),
        constant_pixels=int(np.sum(np.all(train == train[0], axis=0))),
        mean_pair_overlap=float(np.mean(pares)),
        stable_training_patterns=int(estables.sum()), unique_final_states=int(
            len(np.unique(np.array(finals), axis=0))),
        exact_memory_finals=int(sum(np.any(np.all(train == s, axis=1)) for s in finals)),
        total_evaluations=len(rows), capacity_approx=0.138*N, load=70/N,
        python=platform.python_version(), numpy=np.__version__,
        seeds=dict(selection=42, split=43, noise=44), source_url=URL,
        source_sha256=SHA256)
    return rows, resumen, ejemplos, diagnostico


def dibujar(resumen, ejemplos, carpeta):
    """Guarda figuras legibles de confusión, ruido, energía y recuperación."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), layout='constrained')
    for ax, item in zip(axes, resumen):
        cm = np.array(item['confusion_first'])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=3)
        ax.set(title=f"Ruido {item['noise']:.0%}", xlabel='Clase predicha', ylabel='Clase real',
               xticks=range(10), yticks=range(10))
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=8,
                        color='white' if cm[i,j] > 1 else '#44505b')
    fig.savefig(carpeta/'confusion.png', dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4), layout='constrained')
    xx = [x['noise']*100 for x in resumen]
    for pref, label, color in [('accuracy', 'Hopfield + Hamming', '#b44b37'),
                               ('baseline', 'Hamming directo', '#167b81')]:
        yy = [x[pref+'_mean']*100 for x in resumen]
        err = [x[pref+'_std']*100 for x in resumen]
        ax.errorbar(xx, yy, yerr=err, marker='o', capsize=5, label=label, color=color)
    ax.set(xlabel='Bits invertidos nominales (%)', ylabel='Accuracy (%)', ylim=(0,100), xticks=xx)
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(carpeta/'ruido.png', dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4), layout='constrained')
    for q in NIVELES:
        yy = ejemplos[(q, 0)]['cell_energies']
        ax.plot(np.arange(len(yy))/N, yy, label=f'{q:.0%}')
    ax.set(xlabel='Barridos (784 actualizaciones por barrido)', ylabel='Energía E')
    ax.grid(alpha=.2); ax.legend(title='Ruido')
    fig.savefig(carpeta/'energia.png', dpi=220); plt.close(fig)
    fig, axes = plt.subplots(5, 5, figsize=(8.4, 8.2), layout='constrained')
    for row, i in enumerate([0, 3, 6, 12, 24]):
        r=ejemplos[(.10, i)]
        vistas=[r['original'],r['initial'],r['states'][1],r['final'],r['nearest']]
        for col, s in enumerate(vistas):
            axes[row,col].imshow(s.reshape(28,28), cmap='gray', vmin=-1,vmax=1,
                                 interpolation='nearest')
            axes[row,col].set(xticks=[],yticks=[])
            for sp in axes[row,col].spines.values(): sp.set_visible(False)
        axes[row,0].set_ylabel(f"Real {r['true']}\nPred. {r['predicted']}", fontsize=11)
    for ax, title in zip(axes[0], ['Original','Ruido 10%','1 barrido','Estado final','Vecino train']):
        ax.set_title(title, fontsize=10)
    fig.savefig(carpeta/'recuperacion.png', dpi=220); plt.close(fig)


def main():
    """Valida la implementación y genera métricas, trazas y gráficos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerar', action='store_true')
    parser.add_argument('--mnist', type=Path, default=Path(__file__).parent/'cache/mnist.npz')
    parser.add_argument('--salida', type=Path, default=Path(__file__).parent/'resultados')
    parser.add_argument('--repeticiones', type=int, default=20)
    args=parser.parse_args()
    if args.repeticiones < 1: parser.error('--repeticiones debe ser positivo')
    args.salida.mkdir(parents=True, exist_ok=True)
    comprobar_algoritmo()
    datos=preparar_datos(Path(__file__).parent/'mnist_100.npz', args.mnist,args.regenerar)
    rows,resumen,ejemplos,diagnostico=ejecutar(datos,args.repeticiones)
    with (args.salida/'predicciones.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    with (args.salida/'seleccion.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['source_index','label','split'])
        writer.writerows(zip(datos['source_indices'],datos['labels'],
            np.where(datos['train_mask'],'train','test')))
    resultado=dict(summary=resumen,diagnostics=diagnostico)
    (args.salida/'metricas.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    trazas={f'{q:.2f}_{i}':dict(energies=r['energies'],changes=r['changes'],
        cell_energies=r['cell_energies']) for (q,i),r in ejemplos.items()}
    (args.salida/'trazas.json').write_text(json.dumps(trazas),encoding='utf-8')
    np.savez_compressed(args.salida/'ejemplos.npz',**{
        f'{q:.2f}_{i}_{k}':np.array(r[k]) for (q,i),r in ejemplos.items()
        for k in ['original','initial','states','final','nearest']})
    dibujar(resumen,ejemplos,args.salida)
    print(json.dumps(resultado,indent=2))


if __name__ == '__main__':
    main()
