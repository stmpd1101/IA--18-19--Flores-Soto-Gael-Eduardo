import heapq
import random

OBJETIVO = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVS = {'arriba': -3, 'abajo': 3, 'izq': -1, 'der': 1}
COORD = {i: (i // 3, i % 3) for i in range(9)}  

def es_resoluble(estado):
    inversions = 0
    lista = [n for n in estado if n != 0]
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] > lista[j]:
                inversions += 1
    return inversions % 2 == 0

def generar_estado_aleatorio():
    while True:
        estado = tuple(random.sample(range(9), 9))
        if es_resoluble(estado):
            return estado

def funcion_sucesor(estado):
    """Devuelve sucesores válidos al mover el hueco (0) en las 4 direcciones."""
    sucesores = []
    i = estado.index(0)
    for mov, delta in MOVS.items():
        j = i + delta
        # Validaciones de bordes para cada movimiento
        if mov == 'arriba' and i >= 3:
            pass
        elif mov == 'abajo' and i < 6:
            pass
        elif mov == 'izq' and i % 3 != 0:
            pass
        elif mov == 'der' and i % 3 != 2:
            pass
        else:
            continue
        nuevo = list(estado)
        nuevo[i], nuevo[j] = nuevo[j], nuevo[i]
        sucesores.append(tuple(nuevo))
    return sucesores



def dijkstra_puzzle(inicial):
    """Dijkstra para 8-puzzle (cada movimiento cuesta 1).
       Devuelve la lista de estados desde 'inicial' hasta OBJETIVO (inclusive)."""
    if not es_resoluble(inicial):
        return None

    INF = float('inf')
    dist = {inicial: 0}
    parent = {inicial: None}
    heap = [(0, inicial)]               
    visitados = set()

    while heap:
        g, estado = heapq.heappop(heap)
        if estado in visitados:
            continue
        visitados.add(estado)

        if estado == OBJETIVO:         
            break

        for sucesor in funcion_sucesor(estado):
            ng = g + 1                  # cada movimiento cuesta 1
            if ng < dist.get(sucesor, INF):
                dist[sucesor] = ng
                parent[sucesor] = estado
                heapq.heappush(heap, (ng, sucesor))

    if OBJETIVO not in dist:
        return None

    # Reconstrucción de camino
    camino = []
    cur = OBJETIVO
    while cur is not None:
        camino.append(cur)
        cur = parent[cur]
    camino.reverse()
    return camino



if __name__ == "__main__":
    estado_inicial = generar_estado_aleatorio()
    print("Estado inicial:")
    for i in range(0, 9, 3):
        print(estado_inicial[i:i+3])

    solucion = dijkstra_puzzle(estado_inicial)
    if solucion:
        print(f"\nPasos a la solución: {len(solucion)-1}")
        for idx, paso in enumerate(solucion):
            print(f"\nPaso {idx}:")
            for i in range(0, 9, 3):
                print(paso[i:i+3])
    else:
        print("No se encontró solución (o estado no resoluble).")
