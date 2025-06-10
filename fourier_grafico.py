import numpy as np
from matplotlib import use as interfaz_grafica_a_usar
interfaz_grafica_a_usar("QtAgg")
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib.widgets import Slider, Button, RadioButtons

class AnimacionFourier:
    """
    Clase para visualización animadada series de Fourier.
    """
    
    def __init__(self):
        """
        Constructor: Inicializa la aplicación optimizada
        """
        
        # ===== CONFIGURACIÓN DE LA FIGURA PRINCIPAL =====
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('Visualización de Series de Fourier - Composición de Ondas', fontsize=14)
        
        # ===== PARÁMETROS DE LA SIMULACIÓN =====
        self.num_armonicas = 1        # Número actual de armónicas a mostrar
        self.max_armonicas = 50       # Límite máximo de armónicas
        self.tiempo = 0               # Tiempo transcurrido en la animación
        self.dt = 0.025               # Incremento de tiempo por cálculo (mas pequeño, mejor resolucion)
        self.pasos_por_frame = 3      # Número de dt a calcular por frame dibujado
        self.historial_onda = []      # Lista que guarda puntos de la onda para dibujar el trazo
        self.max_historial = 400      # Máximo número de puntos en el historial (ajustar segun dt para mejorar el historial dibujado)
        self.tipo_onda = 'cuadrada'   # Tipo de onda actual
        self.factor_zoom = 1.5        # Factor de zoom para los gráficos
        
        # ===== CONFIGURACIÓN DEL PANEL IZQUIERDO (EPICÍCLOS) =====
        self.ax1.set_xlim(-1.5, 1.5)
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.set_aspect('equal')
        self.ax1.set_title('Epicíclos (Círculos de Fourier)')
        self.ax1.grid(True, alpha=0.3)
        
        # Línea horizontal gruesa en Y=0 para marcar origen del eje X
        self.ax1.axhline(y=0, color='black', linewidth=2, alpha=0.8)
        
        # ===== CONFIGURACIÓN DEL PANEL DERECHO (ONDA RESULTANTE) =====
        self.ax2.set_xlim(0, 4*np.pi)
        self.ax2.set_ylim(-1.5, 1.5)
        self.ax2.set_title('Onda Resultante')
        self.ax2.set_xlabel('Tiempo')
        self.ax2.set_ylabel('Amplitud')
        self.ax2.grid(True, alpha=0.3)
        
        # Línea horizontal gruesa en Y=0 para marcar origen del eje X
        self.ax2.axhline(y=0, color='black', linewidth=2, alpha=0.8)
        
        # ===== INICIALIZACIÓN DE ELEMENTOS GRÁFICOS =====
        self.circulos = []
        self.lineas = []
        self.puntos = []
        
        # Líneas para la onda resultante
        self.punto_actual, = self.ax2.plot([], [], 'ro', markersize=8)
        self.trazo_onda, = self.ax2.plot([], [], 'r-', alpha=0.7, linewidth=1)
        self.linea_referencia = None
        
        # ===== LÍNEA CONECTORA QUE CRUZA POR ENCIMA DE AMBOS PANELES =====
        # Crear línea usando coordenadas de figura (no de axes)
        self.linea_conectora = None  # Se creará dinámicamente
        self.mostrar_conectora = True  # Control para mostrar/ocultar línea
        
        # ===== CONTROL AC/DC =====
        self.modo_dc = False  # False = AC (normal), True = DC (+1 offset)
        
        # Variables para ventanas adicionales
        self.ventana_espectro = None
        self.ventana_componentes = None
        
        # ===== INICIALIZACIÓN FINAL =====
        self.crear_controles()
        self.actualizar_tipo_onda()
        
    def crear_controles(self):
        """
        Crea todos los controles interactivos + control de velocidad
        """
        
        plt.subplots_adjust(bottom=0.3, left=0.15)  # Más espacio para controles adicionales
        
        # ===== SLIDER DE ARMÓNICAS =====
        ax_armonicas = plt.axes([0.25, 0.2, 0.4, 0.03])
        self.slider_armonicas = Slider(
            ax_armonicas, 'Armónicas',
            1, self.max_armonicas,
            valinit=1,
            valfmt='%d'
        )
        self.slider_armonicas.on_changed(self.actualizar_armonicas)
        
        # ===== SLIDER DE ZOOM =====
        ax_zoom = plt.axes([0.25, 0.15, 0.4, 0.03])
        self.slider_zoom = Slider(
            ax_zoom, 'Zoom Out',
            1, 5,
            valinit=1.5,
            valfmt='%.1f'
        )
        self.slider_zoom.on_changed(self.actualizar_zoom)
        
        # ===== SLIDER DE VELOCIDAD =====
        ax_velocidad = plt.axes([0.25, 0.1, 0.4, 0.03])
        self.slider_velocidad = Slider(
            ax_velocidad, 'Velocidad',
            1, 10,                        # 1 = lento, 10 = muy rápido
            valinit=3,                    # Velocidad moderada por defecto
            valfmt='%d'
        )
        self.slider_velocidad.on_changed(self.actualizar_velocidad)
        
        # ===== RADIOBUTTONS PARA TIPO DE ONDA =====
        ax_radio = plt.axes([0.02, 0.35, 0.12, 0.2])
        self.radio_onda = RadioButtons(
            ax_radio, 
            ('Cuadrada', 'Triangular', 'Diente Sierra', 'Senoidal Rectificada')
        )
        self.radio_onda.on_clicked(self.actualizar_tipo_onda_callback)
        
        # ===== BOTONES DE CONTROL =====
        # Fila superior
        ax_play = plt.axes([0.75, 0.2, 0.08, 0.04])
        self.boton_play = Button(ax_play, 'Play/Pause')
        self.boton_play.on_clicked(self.alternar_animacion)
        
        ax_reset = plt.axes([0.85, 0.2, 0.08, 0.04])
        self.boton_reset = Button(ax_reset, 'Reset')
        self.boton_reset.on_clicked(self.reiniciar_animacion)
        
        # Fila inferior
        ax_espectro = plt.axes([0.75, 0.15, 0.08, 0.04])
        self.boton_espectro = Button(ax_espectro, 'Espectro')
        self.boton_espectro.on_clicked(self.mostrar_espectro)
        
        ax_componentes = plt.axes([0.85, 0.15, 0.08, 0.04])
        self.boton_componentes = Button(ax_componentes, 'Ondas')
        self.boton_componentes.on_clicked(self.mostrar_componentes)
        
        # Botón para controlar línea conectora
        ax_conectora = plt.axes([0.75, 0.1, 0.08, 0.04])
        self.boton_conectora = Button(ax_conectora, 'Conectora')
        self.boton_conectora.on_clicked(self.alternar_conectora)
        
        # Botón para intercambiar AC/DC
        ax_acdc = plt.axes([0.85, 0.1, 0.08, 0.04])
        self.boton_acdc = Button(ax_acdc, 'AC/DC')
        self.boton_acdc.on_clicked(self.alternar_ac_dc)
        
        # Variable para controlar animación
        self.esta_jugando = False
        
    def actualizar_velocidad(self, val):
        """
        Callback para el slider de velocidad
        """
        self.pasos_por_frame = int(self.slider_velocidad.val)
        
    def actualizar_armonicas(self, val):
        """Actualizar número de armónicas"""
        self.num_armonicas = int(self.slider_armonicas.val)
        self.actualizar_circulos()
        
    def actualizar_zoom(self, val):
        """Actualizar zoom"""
        self.factor_zoom = val
        limite = self.factor_zoom
        self.ax1.set_ylim(-limite, limite)
        self.ax1.set_xlim(-limite, limite)
        self.ax2.set_ylim(-limite, limite)
        self.fig.canvas.draw_idle()
        
    def actualizar_tipo_onda_callback(self, etiqueta):
        """Callback para cambio de tipo de onda"""
        mapa_onda = {
            'Cuadrada': 'cuadrada', 
            'Triangular': 'triangular', 
            'Diente Sierra': 'sierra', 
            'Senoidal Rectificada': 'rectificada'
        }
        self.tipo_onda = mapa_onda[etiqueta]
        self.actualizar_tipo_onda()
        
    def actualizar_tipo_onda(self):
        """Actualizar configuración completa del tipo de onda"""
        
        nombres_onda = {
            'cuadrada': 'Onda Cuadrada', 
            'triangular': 'Onda Triangular', 
            'sierra': 'Onda Diente de Sierra', 
            'rectificada': 'Senoidal Rectificada'
        }
        
        self.fig.suptitle(f'Visualización Rápida de Series de Fourier - {nombres_onda[self.tipo_onda]}', fontsize=14)
        
        # Crear línea de referencia
        if self.linea_referencia:
            self.linea_referencia.remove()
            
        t_ref = np.linspace(0, 4*np.pi, 1000)
        
        if self.tipo_onda == 'cuadrada':
            onda_ref = np.sign(np.sin(t_ref))
            color = 'g'
        elif self.tipo_onda == 'triangular':
            onda_ref = (2/np.pi) * np.arcsin(np.sin(t_ref))
            color = 'orange'
        elif self.tipo_onda == 'sierra':
            onda_ref = 2 * ((t_ref / (2*np.pi) + 0.5) % 1) - 1
            color = 'purple'
        elif self.tipo_onda == 'rectificada':
            onda_ref = np.abs(np.sin(t_ref))
            color = 'red'
        
        # Agregar componente DC si está activada (excepto para rectificada)
        if self.modo_dc and self.tipo_onda != 'rectificada':
            onda_ref = onda_ref + 1
            
        self.linea_referencia, = self.ax2.plot(
            t_ref, onda_ref, '--',
            color=color, alpha=0.5,
            linewidth=1,
            label=f'{nombres_onda[self.tipo_onda]} ideal'
        )

        self.ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
        
        self.actualizar_circulos()
        self.reiniciar_animacion(None)
        
    def obtener_parametros_armonica(self, indice_armonica):
        """
        Calcular parámetros matemáticos para cada armónica
        """
        
        if self.tipo_onda == 'cuadrada':
            n = 2 * indice_armonica + 1
            amplitud = 4 / (np.pi * n)
            fase = 0
            frecuencia = n
            
        elif self.tipo_onda == 'triangular':
            n = 2 * indice_armonica + 1
            amplitud = 8 / (np.pi**2 * n**2)
            fase = 0 if indice_armonica % 2 == 0 else np.pi
            frecuencia = n
            
        elif self.tipo_onda == 'sierra':
            n = indice_armonica + 1
            amplitud = 2 / (np.pi * n)
            fase = np.pi if n % 2 == 0 else 0
            frecuencia = n
            
        elif self.tipo_onda == 'rectificada':
            if indice_armonica == 0:
                amplitud = 2 / np.pi
                fase = 0
                frecuencia = 0
            else:
                n = 2 * indice_armonica
                amplitud = -4 / (np.pi * (1 - n**2))
                fase = -np.pi/2
                frecuencia = n
            
        return amplitud, fase, frecuencia
        
    def actualizar_circulos(self):
        """
        Actualizar elementos gráficos de epicíclos
        """
        
        for circulo in self.circulos:
            circulo.remove()
        for linea in self.lineas:
            linea.remove()
        for punto in self.puntos:
            punto.remove()
            
        self.circulos = []
        self.lineas = []
        self.puntos = []
        
        # Agregar elemento visual para componente DC si está activa
        if self.modo_dc and self.tipo_onda != 'rectificada':
            # Círculo rojo para mostrar el "epicíclo" de la componente DC
            circulo_dc = plt.Circle(
                (0, 0),  # Centrado en el origen
                1,  # Radio = 1 (amplitud de la componente DC)
                fill=False,
                color='red',
                alpha=0.6,
                linewidth=2,
                linestyle='--'  # Línea punteada para diferenciarlo
            )
            self.ax1.add_patch(circulo_dc)
            self.circulos.append(circulo_dc)
            
            # Línea vertical para mostrar componente DC
            linea_dc, = self.ax1.plot([0, 0], [0, 1], color='red', linewidth=3, alpha=0.8)
            self.lineas.append(linea_dc)
            
            # Punto para componente DC
            punto_dc, = self.ax1.plot([0], [1], 's', color='red', markersize=8, alpha=0.8)
            self.puntos.append(punto_dc)
        
        # Crear círculos normales para las armónicas
        for i in range(self.num_armonicas):
            amplitud, fase, frecuencia = self.obtener_parametros_armonica(i)
            
            # Para onda rectificada, la primera armónica (frecuencia=0) es DC
            if self.tipo_onda == 'rectificada' and frecuencia == 0:
                # Crear círculo DC rojo punteado para rectificada
                circulo = plt.Circle(
                    (0, 0),  # Centrado en origen 
                    amplitud,
                    fill=False,
                    color='red',  # Rojo en lugar de azul
                    alpha=0.6,
                    linewidth=2,
                    linestyle='--'  # Punteado en lugar de sólido
                )
            else:
                # Círculo normal para otras armónicas
                circulo = plt.Circle(
                    (0, 0),  # Se centrará correctamente en animar()
                    amplitud,
                    fill=False,
                    color=f'C{i}',
                    alpha=0.7
                )
                
            self.ax1.add_patch(circulo)
            self.circulos.append(circulo)
            
            # Determinar color de línea y punto según tipo de onda y frecuencia
            if self.tipo_onda == 'rectificada' and frecuencia == 0:
                # Para componente DC de señal rectificada: usar color rojo
                color_linea = 'red'
                color_punto = 'red'
            else:
                # Para armónicas normales: usar color estándar
                color_linea = f'C{i}'
                color_punto = f'C{i}'
            
            linea, = self.ax1.plot([], [], color=color_linea, linewidth=2)
            self.lineas.append(linea)
            
            punto, = self.ax1.plot([], [], 'o', color=color_punto, markersize=6)
            self.puntos.append(punto)
        
    def calcular_punto_fourier(self, t):
        """
        Algoritmo principal de suma de Fourier
        """
        
        x, y = 0, 0
        posiciones = [(0, 0)]
        
        # Agregar componente DC si está activada (excepto para rectificada)
        if self.modo_dc and self.tipo_onda != 'rectificada':
            y += 1
            posiciones.append((0, 1))  # Punto DC fijo en (0, 1)
        
        for i in range(self.num_armonicas):
            amplitud, fase, frecuencia = self.obtener_parametros_armonica(i)
                
            if frecuencia == 0:
                dx = 0
                dy = amplitud
            else:
                angulo = frecuencia * t + fase
                dx = amplitud * np.cos(angulo)
                dy = amplitud * np.sin(angulo)
            
            x += dx
            y += dy
            posiciones.append((x, y))
        
        return posiciones, y
        
    def animar(self, frame):
        """
        Función de animación
        """
        
        if not self.esta_jugando:
            return []
            
        # ===== CÁLCULO DE MÚLTIPLES PASOS =====
        nuevos_puntos = []  # Lista para acumular puntos calculados
        posiciones_finales = None
        y_final = 0
        
        for paso in range(self.pasos_por_frame):
            # Avanzar tiempo
            self.tiempo += self.dt
            
            # Calcular nueva posición
            posiciones, val_y = self.calcular_punto_fourier(self.tiempo)
            posiciones_finales = posiciones
            y_final = val_y
            
            # Acumular punto para el trazado
            nuevos_puntos.append((self.tiempo, val_y))
            
            # Solo actualizar epicíclos en el último paso (para visualización)
            if paso == self.pasos_por_frame - 1:
                # ===== ACTUALIZAR EPICÍCLOS =====
                
                # Si hay DC, hay UN elemento extra en lineas y UN elemento extra en puntos
                desplazamiento_dc = 1 if (self.modo_dc and self.tipo_onda != 'rectificada') else 0
                desplazamiento_circulo_dc = 1 if (self.modo_dc and self.tipo_onda != 'rectificada') else 0
                
                for i in range(self.num_armonicas):
                    # Los círculos tienen offset por círculo DC
                    indice_circulo = i + desplazamiento_circulo_dc
                    
                    # Las líneas y puntos tienen offset por elementos DC
                    indice_linea = i + desplazamiento_dc
                    indice_punto = i + desplazamiento_dc
                    
                    # En posiciones: [0]=(0,0), [1]=DC si existe, luego armónicas
                    if self.modo_dc and self.tipo_onda != 'rectificada':
                        indice_pos_centro = i + 1  # Primera armónica se centra en DC posicion[1]
                        indice_pos_fin = i + 2     # Y va hasta posicion[2]
                    else:
                        indice_pos_centro = i      # Primera armónica se centra en posicion[0]=(0,0)
                        indice_pos_fin = i + 1     # Y va hasta posicion[1]
                    
                    if (indice_circulo < len(self.circulos) and 
                        indice_linea < len(self.lineas) and 
                        indice_pos_fin < len(posiciones)):
                        
                        # Centrar círculo en la posición correcta
                        posicion_centro = posiciones[indice_pos_centro]
                        self.circulos[indice_circulo].center = posicion_centro
                        
                        # Dibujar línea desde centro hasta extremo
                        x_inicio, y_inicio = posiciones[indice_pos_centro]
                        x_fin, y_fin = posiciones[indice_pos_fin]
                        self.lineas[indice_linea].set_data([x_inicio, x_fin], [y_inicio, y_fin])
                        
                        # Colocar punto en el extremo
                        self.puntos[indice_punto].set_data([x_fin], [y_fin])
                
                # Actualizar punto rojo actual
                self.punto_actual.set_data([self.tiempo], [y_final])
        
        # ===== GESTIÓN DE HISTORIAL CON MÚLTIPLES PUNTOS =====
        # Añadir todos los puntos calculados al historial
        self.historial_onda.extend(nuevos_puntos)
        
        # Limitar tamaño del historial
        while len(self.historial_onda) > self.max_historial:
            self.historial_onda.pop(0)
            
        # ===== ACTUALIZAR TRAZADO DE ONDA =====
        if self.historial_onda:
            vals_t, vals_y = zip(*self.historial_onda)
            self.trazo_onda.set_data(vals_t, vals_y)
        
        # ===== DESPLAZAMIENTO AUTOMÁTICO DE VENTANA =====
        eje_desplazado = False
        if self.tiempo > self.ax2.get_xlim()[1] - np.pi:
            self.ax2.set_xlim(self.tiempo - 3*np.pi, self.tiempo + np.pi)
            eje_desplazado = True
            
            # ===== ACTUALIZAR ONDA IDEAL CUANDO SE DESPLAZA EL EJE =====
            if self.linea_referencia:
                # Rango visible actual del eje
                xlim_actual = self.ax2.get_xlim()
                t_ref = np.linspace(xlim_actual[0], xlim_actual[1], 1000)
                
                # Recalcular onda ideal para el rango visible
                if self.tipo_onda == 'cuadrada':
                    onda_ref = np.sign(np.sin(t_ref))
                elif self.tipo_onda == 'triangular':
                    onda_ref = (2/np.pi) * np.arcsin(np.sin(t_ref))
                elif self.tipo_onda == 'sierra':
                    onda_ref = 2 * ((t_ref / (2*np.pi) + 0.5) % 1) - 1
                elif self.tipo_onda == 'rectificada':
                    onda_ref = np.abs(np.sin(t_ref))
                
                # Agregar componente DC si está activada (excepto para rectificada)
                if self.modo_dc and self.tipo_onda != 'rectificada':
                    onda_ref = onda_ref + 1
                
                # Actualizar datos de la línea de referencia
                self.linea_referencia.set_data(t_ref, onda_ref)
        
        # ===== ACTUALIZAR LÍNEA CONECTORA DESPUÉS DEL DESPLAZAMIENTO =====
        # Recalcular siempre después de cualquier cambio en el eje
        if posiciones_finales and len(posiciones_finales) > 0 and self.mostrar_conectora:
            # El último punto de los epicíclos está siempre en posiciones_finales[-1]
            epiciclo_x, epiciclo_y = posiciones_finales[-1]
            display_epiciclo = self.ax1.transData.transform([[epiciclo_x, epiciclo_y]])[0]
            fig_epiciclo = self.fig.transFigure.inverted().transform(display_epiciclo)
            
            # Coordenadas del punto rojo actual (que ya está correcto)
            datos_punto_actual = self.punto_actual.get_data()
            if len(datos_punto_actual[0]) > 0 and len(datos_punto_actual[1]) > 0:
                onda_x = datos_punto_actual[0][0]  # Usar coordenadas del punto rojo
                onda_y = datos_punto_actual[1][0]
                
                # Convertir coordenadas actualizadas a coordenadas de figura
                display_onda = self.ax2.transData.transform([[onda_x, onda_y]])[0]
                fig_onda = self.fig.transFigure.inverted().transform(display_onda)
                
                # Actualizar o crear línea
                if self.linea_conectora is None:
                    self.linea_conectora = plt.Line2D([fig_epiciclo[0], fig_onda[0]], 
                                                   [fig_epiciclo[1], fig_onda[1]], 
                                                   color='green', linewidth=2, alpha=0.5,
                                                   transform=self.fig.transFigure)
                    self.fig.lines.append(self.linea_conectora)
                else:
                    self.linea_conectora.set_data([fig_epiciclo[0], fig_onda[0]], 
                                               [fig_epiciclo[1], fig_onda[1]])
        elif not self.mostrar_conectora and self.linea_conectora is not None:
            self.linea_conectora.set_data([], [])
        
        return self.lineas + self.puntos + [self.trazo_onda, self.punto_actual]
        
    def alternar_animacion(self, evento):
        """Alternar play/pause"""
        self.esta_jugando = not self.esta_jugando
        
    def reiniciar_animacion(self, evento):
        """Reiniciar animación"""
        self.tiempo = 0
        self.historial_onda = []
        self.trazo_onda.set_data([], [])
        self.punto_actual.set_data([], [])
        self.ax2.set_xlim(0, 4*np.pi)
        
        # Limpiar línea conectora
        if self.linea_conectora is not None:
            self.linea_conectora.set_data([], [])
        
    def alternar_conectora(self, evento):
        """
        Alternar visibilidad de línea conectora
        """
        self.mostrar_conectora = not self.mostrar_conectora
        if not self.mostrar_conectora and self.linea_conectora is not None:
            self.linea_conectora.set_data([], [])
        
    def alternar_ac_dc(self, evento):
        """
        Alternar entre AC y DC
        """
        
        # No hacer nada si la señal es rectificada
        if self.tipo_onda == 'rectificada':
            return
            
        self.modo_dc = not self.modo_dc
        
        # Ajustar zoom automáticamente según modo AC/DC
        if self.modo_dc:
            # Modo DC: zoom por defecto 2.5
            self.slider_zoom.set_val(2.5)
        else:
            # Modo AC: zoom por defecto 1.5
            self.slider_zoom.set_val(1.5)
        
        # Actualizar el tipo de onda para refrescar la referencia
        self.actualizar_tipo_onda()
        
    def iniciar(self):
        """
        Iniciar la aplicación
        """
        # Intervalo más alto para compensar múltiples cálculos por frame
        self.anim = animation.FuncAnimation(
            self.fig,
            self.animar,
            interval=16,  # ~60 FPS para compensar múltiples pasos
            blit=False,
            repeat=True
        )
        plt.show()

    def mostrar_espectro(self, evento):
        """
        Análisis espectral
        """
        
        if self.ventana_espectro is not None:
            plt.close(self.ventana_espectro)
            
        self.ventana_espectro = plt.figure(figsize=(8, 6))
        self.ventana_espectro.suptitle(f'Análisis Espectral - {self.obtener_nombre_onda()}', fontsize=14)
        
        ax_espectro = self.ventana_espectro.add_subplot(1, 1, 1)
        
        frecuencias = []
        amplitudes = []
        amplitud_fundamental = 0
        
        for i in range(self.num_armonicas):
            amplitud, fase, frecuencia = self.obtener_parametros_armonica(i)
            if amplitud > 0.001:
                if amplitud > amplitud_fundamental:
                    amplitud_fundamental = amplitud
        
        cuenta_significativas = 0
        for i in range(self.num_armonicas):
            amplitud, fase, frecuencia = self.obtener_parametros_armonica(i)
            if amplitud > 0.001:
                frecuencias.append(frecuencia)
                amplitudes.append(amplitud)
                
                if amplitud > 0.05 * amplitud_fundamental:
                    cuenta_significativas += 1
        
        linea_marcador, lineas_tallo, linea_base = ax_espectro.stem(frecuencias, amplitudes, basefmt='b-')
        plt.setp(linea_marcador, markersize=8, color='red')
        plt.setp(lineas_tallo, linewidth=2, color='blue')
        
        ax_espectro.set_xlabel('Frecuencia (armónica)', fontsize=12)
        ax_espectro.set_ylabel('Amplitud', fontsize=12)
        ax_espectro.set_title('Espectro de Frecuencias', fontsize=14)
        ax_espectro.grid(True, alpha=0.3)
        ax_espectro.set_xlim(-0.5, max(frecuencias) + 0.5 if frecuencias else 1)
        
        for frec, amp in zip(frecuencias, amplitudes):
            ax_espectro.text(
                frec, amp + max(amplitudes)*0.02,
                f'{amp:.3f}',
                ha='center', va='bottom',
                fontsize=10
            )
        
        texto_info = f"Armónicas analizadas: {self.num_armonicas}\n"
        texto_info += f"Componentes significativas (>5% fundamental): {cuenta_significativas}"
        ax_espectro.text(
            0.02, 0.98,
            texto_info,
            transform=ax_espectro.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        plt.show()
        
    def mostrar_componentes(self, evento):
        """
        Visualización de componentes individuales
        """
        
        if self.ventana_componentes is not None:
            plt.close(self.ventana_componentes)
            
        self.ventana_componentes = plt.figure(figsize=(10, 8))
        self.ventana_componentes.suptitle(f'Ondas Senoidales Individuales - {self.obtener_nombre_onda()}', fontsize=14)
        
        frecuencias = []
        amplitudes = []
        fases = []
        
        for i in range(self.num_armonicas):
            amplitud, fase, frecuencia = self.obtener_parametros_armonica(i)
            if amplitud > 0.05:
                frecuencias.append(frecuencia)
                amplitudes.append(amplitud)
                fases.append(fase)
        
        ax_componentes = self.ventana_componentes.add_subplot(1, 1, 1)
        
        max_amplitud = max(amplitudes) if amplitudes else 1
        espaciado = 2.0 * max_amplitud
        desplazamiento_y = -1.0 * max_amplitud
        
        t = np.linspace(0, 4*np.pi, 1000)
        colores = plt.cm.tab10(np.linspace(0, 1, len(frecuencias)))
        
        for i, (frec, amp, fase, color) in enumerate(zip(frecuencias, amplitudes, fases, colores)):
            if frec == 0:
                y_individual = np.ones_like(t) * amp
                etiqueta = f'DC (A={amp:.3f})'
            else:
                y_individual = amp * np.sin(frec * t + fase)
                etiqueta = f'Armónica {int(frec)} (A={amp:.3f})'
            
            y_grafico = y_individual + desplazamiento_y
            
            ax_componentes.plot(t, y_grafico, color=color, linewidth=2, label=etiqueta)
            ax_componentes.axhline(y=desplazamiento_y, color=color, alpha=0.3, linestyle='--', linewidth=1)
            
            ax_componentes.text(
                -0.5, desplazamiento_y,
                f'#{int(frec)}',
                verticalalignment='center',
                fontsize=10,
                color=color,
                fontweight='bold'
            )
            
            desplazamiento_y -= espaciado
        
        ax_componentes.set_xlabel('Tiempo', fontsize=12)
        ax_componentes.set_title('Ondas Senoidales Que Componen La Señal', fontsize=14)
        ax_componentes.grid(True, alpha=0.3)
        ax_componentes.legend(loc='upper right', fontsize=9)
        ax_componentes.set_xlim(0, 4*np.pi)
        ax_componentes.set_yticks([])
        
        y_superior = max_amplitud * 0.5
        y_inferior = desplazamiento_y - max_amplitud * 0.3
        ax_componentes.set_ylim(y_inferior, y_superior)
        
        plt.tight_layout()
        plt.show()
        
    def obtener_nombre_onda(self):
        """
        Obtener nombre de la onda actual
        """
        nombres_onda = {
            'cuadrada': 'Onda Cuadrada', 
            'triangular': 'Onda Triangular', 
            'sierra': 'Onda Diente de Sierra', 
            'rectificada': 'Senoidal Rectificada'
        }
        return nombres_onda.get(self.tipo_onda, 'Onda Desconocida')

# ===== PUNTO DE ENTRADA DEL PROGRAMA =====
if __name__ == "__main__":
    """
    Función principal optimizada para velocidad
    """

    viz_fourier = AnimacionFourier()
    viz_fourier.iniciar() 