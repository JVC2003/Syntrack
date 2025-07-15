import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import json
import os

from dataset import AudioMatchingDataset
from similarity_matching_model import SimilarityMatchingModel, EnhancedSimilarityModel
from improved_utils import collate_fn, calculate_metrics, adaptive_threshold_selection

class ModelEvaluator:
    def __init__(self, model_path, dataset_path, clips_dir, songs_dir, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dataset_path = dataset_path
        self.clips_dir = clips_dir
        self.songs_dir = songs_dir
        
        # Cargar modelo
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Crear modelo
        if self.config['model_type'] == 'enhanced':
            self.model = EnhancedSimilarityModel()
        else:
            self.model = SimilarityMatchingModel()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Modelo cargado desde: {model_path}")
        print(f"Mejor MAE durante entrenamiento: {checkpoint['best_mae']:.4f}")
    
    def setup_test_loader(self):
        """Configura el data loader para evaluación"""
        dataset = AudioMatchingDataset(
            excel_path=self.dataset_path,
            clips_dir=self.clips_dir,
            songs_dir=self.songs_dir
        )
        
        # Usar mismo split que en entrenamiento
        by_song = defaultdict(list)
        for idx, sample in enumerate(dataset):
            song_key = sample['clip_id'].split("_clip")[0]
            by_song[song_key].append(idx)
        
        all_keys = list(by_song.keys())
        
        # Test set: últimos 15% de canciones
        test_songs = all_keys[int(0.85 * len(all_keys)):]
        test_indices = [i for song in test_songs for i in by_song[song]]
        
        test_set = Subset(dataset, test_indices)
        test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_fn)
        
        print(f"Set de evaluación: {len(test_indices)} clips de {len(test_songs)} canciones")
        return test_loader
    
    def predict_batch(self, test_loader):
        """Realiza predicciones en todo el dataset de test"""
        predictions = []
        targets = []
        confidences = []
        clip_ids = []
        song_lengths = []
        errors_seconds = []
        
        with torch.no_grad():
            for batch in test_loader:
                if not batch['valid'][0]:
                    continue
                
                clip = batch['clip'][0].unsqueeze(0).to(self.device)
                song = batch['song'][0].unsqueeze(0).to(self.device)
                target = batch['target'].to(self.device)
                clip_id = batch['clip_id'][0]
                
                # Predicción
                ts_pred, conf_pred = self.model(clip, song)
                
                # Convertir a segundos (aproximado)
                song_length_frames = song.shape[1]
                song_length_seconds = song_length_frames / 10  # 10 fps
                
                pred_seconds = ts_pred.item() * song_length_seconds
                target_seconds = target.item() * song_length_seconds
                error_seconds = abs(pred_seconds - target_seconds)
                
                # Almacenar resultados
                predictions.append(ts_pred.cpu().item())
                targets.append(target.cpu().item())
                confidences.append(conf_pred.cpu().item())
                clip_ids.append(clip_id)
                song_lengths.append(song_length_seconds)
                errors_seconds.append(error_seconds)
        
        return {
            'predictions': np.array(predictions),
            'targets': np.array(targets),
            'confidences': np.array(confidences),
            'clip_ids': clip_ids,
            'song_lengths': np.array(song_lengths),
            'errors_seconds': np.array(errors_seconds)
        }
    
    def calculate_detailed_metrics(self, results):
        """Calcula métricas detalladas"""
        predictions = torch.tensor(results['predictions'])
        targets = torch.tensor(results['targets'])
        confidences = torch.tensor(results['confidences'])
        errors_seconds = results['errors_seconds']
        
        # Métricas básicas
        metrics = calculate_metrics(predictions, targets, confidences, 
                                  time_thresholds=[1, 2, 3, 5, 10, 15])
        
        # Métricas adicionales en segundos
        metrics['mae_seconds'] = np.mean(errors_seconds)
        metrics['median_error_seconds'] = np.median(errors_seconds)
        metrics['std_error_seconds'] = np.std(errors_seconds)
        metrics['max_error_seconds'] = np.max(errors_seconds)
        metrics['min_error_seconds'] = np.min(errors_seconds)
        
        # Percentiles de error
        for p in [50, 75, 90, 95, 99]:
            metrics[f'error_p{p}_seconds'] = np.percentile(errors_seconds, p)
        
        # Análisis por rangos de confianza
        conf_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in conf_ranges:
            mask = (confidences >= low) & (confidences < high)
            if mask.sum() > 0:
                range_errors = errors_seconds[mask]
                metrics[f'mae_conf_{low}-{high}'] = np.mean(range_errors)
                metrics[f'count_conf_{low}-{high}'] = mask.sum().item()
        
        # Umbral óptimo de confianza
        errors_norm = torch.abs(predictions - targets)
        optimal_threshold = adaptive_threshold_selection(
            confidences, errors_norm, target_precision=0.8
        )
        metrics['optimal_confidence_threshold'] = optimal_threshold
        
        # Métricas con umbral óptimo
        high_conf_mask = confidences >= optimal_threshold
        if high_conf_mask.sum() > 0:
            metrics['high_conf_mae_seconds'] = np.mean(errors_seconds[high_conf_mask])
            metrics['high_conf_fraction'] = high_conf_mask.sum().item() / len(confidences)
            
            # Recall para casos de alta confianza
            for threshold_seconds in [1, 3, 5]:
                recall = np.mean(errors_seconds[high_conf_mask] < threshold_seconds)
                metrics[f'high_conf_recall_{threshold_seconds}s'] = recall
        
        return metrics
    
    def generate_visualizations(self, results, save_dir='evaluation_plots'):
        """Genera visualizaciones de los resultados"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución de errores
        axes[0,0].hist(results['errors_seconds'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Error (segundos)')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].set_title('Distribución de Errores Temporales')
        axes[0,0].axvline(np.median(results['errors_seconds']), color='red', 
                         linestyle='--', label=f'Mediana: {np.median(results["errors_seconds"]):.1f}s')
        axes[0,0].legend()
        
        # 2. Correlación predicción vs target
        axes[0,1].scatter(results['targets'], results['predictions'], 
                         c=results['confidences'], cmap='viridis', alpha=0.6)
        axes[0,1].plot([0, 1], [0, 1], 'r--', label='Predicción perfecta')
        axes[0,1].set_xlabel('Target (normalizado)')
        axes[0,1].set_ylabel('Predicción (normalizado)')
        axes[0,1].set_title('Predicción vs Target')
        axes[0,1].legend()
        cbar = plt.colorbar(axes[0,1].collections[0], ax=axes[0,1])
        cbar.set_label('Confianza')
        
        # 3. Error vs Confianza
        axes[0,2].scatter(results['confidences'], results['errors_seconds'], alpha=0.6)
        axes[0,2].set_xlabel('Confianza')
        axes[0,2].set_ylabel('Error (segundos)')
        axes[0,2].set_title('Error vs Confianza')
        
        # Línea de tendencia
        z = np.polyfit(results['confidences'], results['errors_seconds'], 1)
        p = np.poly1d(z)
        axes[0,2].plot(results['confidences'], p(results['confidences']), "r--", alpha=0.8)
        
        # 4. Distribución de confianzas
        axes[1,0].hist(results['confidences'], bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Confianza')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_title('Distribución de Confianzas')
        
        # 5. Recall por threshold
        thresholds = [1, 2, 3, 5, 10, 15, 20]
        recalls = [np.mean(results['errors_seconds'] < t) for t in thresholds]
        axes[1,1].plot(thresholds, recalls, 'bo-', linewidth=2, markersize=8)
        axes[1,1].set_xlabel('Threshold (segundos)')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].set_title('Recall vs Threshold Temporal')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Boxplot de errores por rango de confianza
        conf_bins = np.digitize(results['confidences'], bins=np.linspace(0, 1, 6))
        conf_data = []
        conf_labels = []
        for i in range(1, 6):
            mask = conf_bins == i
            if np.sum(mask) > 0:
                conf_data.append(results['errors_seconds'][mask])
                conf_labels.append(f'{(i-1)*0.2:.1f}-{i*0.2:.1f}')
        
        if conf_data:
            axes[1,2].boxplot(conf_data, labels=conf_labels)
            axes[1,2].set_xlabel('Rango de Confianza')
            axes[1,2].set_ylabel('Error (segundos)')
            axes[1,2].set_title('Error por Rango de Confianza')
            axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_failure_cases(self, results, top_k=10):
        """Analiza los casos con mayor error"""
        errors = results['errors_seconds']
        indices = np.argsort(errors)[-top_k:]  # Top K errores
        
        print(f"\n{'='*60}")
        print(f"ANÁLISIS DE {top_k} PEORES CASOS")
        print(f"{'='*60}")
        
        failure_analysis = []
        for i, idx in enumerate(reversed(indices)):
            clip_id = results['clip_ids'][idx]
            error = errors[idx]
            confidence = results['confidences'][idx]
            song_length = results['song_lengths'][idx]
            pred_time = results['predictions'][idx] * song_length
            target_time = results['targets'][idx] * song_length
            
            analysis = {
                'rank': i + 1,
                'clip_id': clip_id,
                'error_seconds': error,
                'confidence': confidence,
                'song_length': song_length,
                'predicted_time': pred_time,
                'target_time': target_time,
                'error_percentage': (error / song_length) * 100
            }
            
            failure_analysis.append(analysis)
            
            print(f"\n{i+1}. {clip_id}")
            print(f"   Error: {error:.1f}s ({(error/song_length)*100:.1f}% de la canción)")
            print(f"   Confianza: {confidence:.3f}")
            print(f"   Predicho: {pred_time:.1f}s | Real: {target_time:.1f}s")
            print(f"   Duración canción: {song_length:.1f}s")
        
        return failure_analysis
    
    def generate_report(self, results, metrics, save_path='evaluation_report.json'):
        """Genera reporte completo de evaluación"""
        
        failure_cases = self.analyze_failure_cases(results)
        
        report = {
            'model_config': self.config,
            'evaluation_summary': {
                'total_clips': len(results['clip_ids']),
                'mean_song_length': float(np.mean(results['song_lengths'])),
                'std_song_length': float(np.std(results['song_lengths']))
            },
            'metrics': {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                       for k, v in metrics.items()},
            'failure_analysis': failure_cases
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReporte guardado en: {save_path}")
        return report
    
    def evaluate(self, generate_plots=True, analyze_failures=True):
        """Evaluación completa del modelo"""
        print("Iniciando evaluación...")
        
        # Configurar datos
        test_loader = self.setup_test_loader()
        
        # Realizar predicciones
        print("Realizando predicciones...")
        results = self.predict_batch(test_loader)
        
        # Calcular métricas
        print("Calculando métricas...")
        metrics = self.calculate_detailed_metrics(results)
        
        # Mostrar resumen
        print(f"\n{'='*60}")
        print("RESUMEN DE EVALUACIÓN")
        print(f"{'='*60}")
        print(f"Total de clips evaluados: {len(results['clip_ids'])}")
        print(f"MAE (segundos): {metrics['mae_seconds']:.2f}")
        print(f"Mediana de error: {metrics['median_error_seconds']:.2f}s")
        print(f"Error máximo: {metrics['max_error_seconds']:.1f}s")
        print(f"Confianza promedio: {metrics['mean_confidence']:.3f}")
        print(f"Correlación confianza-precisión: {metrics.get('confidence_accuracy_correlation', 'N/A')}")
        
        print(f"\nRecall por threshold:")
        for key, value in metrics.items():
            if 'recall_at' in key and 'high_conf' not in key:
                print(f"  {key}: {value:.1%}")
        
        print(f"\nUmbral óptimo de confianza: {metrics['optimal_confidence_threshold']:.3f}")
        if 'high_conf_fraction' in metrics:
            print(f"Fracción con alta confianza: {metrics['high_conf_fraction']:.1%}")
            print(f"MAE para alta confianza: {metrics.get('high_conf_mae_seconds', 'N/A'):.2f}s")
        
        # Generar visualizaciones
        if generate_plots:
            print("\nGenerando visualizaciones...")
            self.generate_visualizations(results)
        
        # Analizar casos fallidos
        if analyze_failures:
            self.analyze_failure_cases(results)
        
        # Generar reporte
        report = self.generate_report(results, metrics)
        
        return results, metrics, report

def main():
    evaluator = ModelEvaluator(
        model_path='models/best_model.pth',
        dataset_path='clips_syntrack/super_etiquetado_Wowow.xlsx',
        clips_dir='embeddings/clips',
        songs_dir='embeddings/songs'
    )
    
    results, metrics, report = evaluator.evaluate()

if __name__ == "__main__":
    main()