import os
import torch
import sys

class OpenVoiceToneConverter:
    def __init__(self, model_dir="./models/openvoice", device=None):
        self.model_dir = model_dir
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.config_path = os.path.join(self.model_dir, "config.json")
        self.checkpoint_path = os.path.join(self.model_dir, "converter.pth")
        self.converter = None
        
    def _load_converter(self):
        if self.converter is not None:
            return
            
        if not os.path.exists(self.config_path) or not os.path.exists(self.checkpoint_path):
            print("\n" + "="*60)
            print("❌ ERREUR : Les poids du modèle OpenVoice (Tone Color Converter) sont manquants.")
            print(f"   Dossier attendu : {self.model_dir}")
            print("👉 Pour installer OpenVoice et télécharger les poids, lancez :")
            print("   pip install openvoice-cli --break-system-packages")
            print("   mkdir -p models/openvoice")
            print("   wget -O models/openvoice/converter.pth https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/v2/converter.pth")
            print("   wget -O models/openvoice/config.json https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/v2/config.json")
            print("="*60 + "\n")
            raise FileNotFoundError("Poids de modèle OpenVoice introuvables.")
            
        try:
            from openvoice.api import ToneColorConverter
            self.converter = ToneColorConverter(self.config_path, device=self.device)
            self.converter.load_ckpt(self.checkpoint_path)
        except ImportError as e:
            print("\n" + "="*60)
            print("❌ ERREUR : La bibliothèque openvoice-cli n'est pas installée.")
            print(f"   Détail: {e}")
            print("👉 Installez-la en lançant :")
            print("   pip install openvoice-cli --break-system-packages")
            print("="*60 + "\n")
            raise e

    def convert_timbre(self, synthesized_wav: str, reference_wav: str, output_wav: str):
        self._load_converter()
        
        from openvoice import se_extractor
        
        # Extraction de l'empreinte vocale source (voix de synthèse)
        source_se, _ = se_extractor.get_se(synthesized_wav, self.converter, vad=False)
        # Extraction de l'empreinte vocale cible (voix originale de référence)
        target_se, _ = se_extractor.get_se(reference_wav, self.converter, vad=False)
        
        # Conversion du timbre (Tone Color Conversion)
        self.converter.convert(
            src_se=source_se,
            tgt_se=target_se,
            src_path=synthesized_wav,
            save_path=output_wav
        )
