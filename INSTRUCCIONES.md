
### 🧪 **Instrucciones para correr el script de factibilidad de transpile**


#### 📦 1. Instala Poetry (si aún no está instalado)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Luego agrega esto al `~/.bashrc` o `~/.zshrc` si es necesario:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Aplica los cambios:

```bash
source ~/.bashrc
```

---

#### 📁 2. Instala las dependencias del proyecto

```bash
poetry install
```

---

#### 🚀 3. Ejecuta el script para correr los benchmarks

```bash
poetry run ./pre_eval_run.py
```

---

#### 📊 4. Si ya tienes resultados y solo quieres generar el reporte

```bash
poetry run ./pre_eval_run.py --report-only
```

El primer run está configurado para probar sudokus 4x4 con 2 hasta 15 celdas faltantes:

```python
SUBGRID_SIZES: List[int]  = [2]       # test 4×4
MISSING_CELLS: List[int]  = list(range(2, 16))  # Number of missing cells to test
SAMPLES_PER_COMBO: int    = 3 # How many samples to take per (size, missing) combo
```

Luego hay que probar esta configuración (porque hay un solo 4x4 con 16 celdas restantes)

```python
SUBGRID_SIZES: List[int]  = [2]
MISSING_CELLS: List[int]  = [16]
SAMPLES_PER_COMBO: int    = 1

CSV_PATH          = Path("transpile_feasibility_2.csv")
```

Y luego esta:

(Creo que unas 35 celdas restantes ya es demasiado pero a ver si funciona)

```python
SUBGRID_SIZES: List[int]  = [3]       # test 9×9
MISSING_CELLS: List[int]  = list(range(2, 55))
SAMPLES_PER_COMBO: int    = 3
```
---

### 📝 Archivos generados

* Resultados CSV: `transpile_feasibility_{SUBGRID_SIZES}_{SAMPLES_PER_COMBO}.csv`
* Reporte JSON: `feasibility_report_{SUBGRID_SIZES}_{SAMPLES_PER_COMBO}.json`
* Resumen impreso en consola
