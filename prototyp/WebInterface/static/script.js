document.addEventListener('DOMContentLoaded', () => {
  // Dynamische Galerieaktualisierung beim Hochladen von Bildern
  const uploadForm = document.getElementById('uploadForm');

  uploadForm.addEventListener('submit', event => {
    event.preventDefault(); // Verhindere die Standardaktion des Formulars

    const fileInput = document.getElementById('fileInput');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]); // Füge die ausgewählte Datei hinzu

    // Lade das Bild hoch und aktualisiere die Galerie
    fetch('/upload', {
      method: 'POST',
      body: formData
    })
        .then(response => {
          if (!response.ok) {
            return response.text().then(error => { throw new Error(error); });
          }
          return response.json();
        })
        .then(data => {
          // Zeige das neue Bild in der Galerie
          const gallery = document.querySelector('.d-flex'); // Ziel-Galerie
          const originalImage = data.original;
          const processedImage = data.processed;

          gallery.innerHTML += `
          <div class="zoom-container m-2" style="width: 200px; height: 200px;">
            <img src="${processedImage}" class="img-thumbnail" alt="Verarbeitetes Bild">
          </div>
        `;
          // Erfolgsnachricht anzeigen
          document.getElementById('result').innerHTML = `<p class="text-success">Bild erfolgreich hochgeladen!</p>`;
        })
        .catch(error => {
          // Zeige die Fehlermeldung an
          document.getElementById('result').innerHTML = `<p class="text-danger">Fehler: ${error.message}</p>`;
        });
  });

  // Navigation Buttons und Fortschrittsbalken (falls vorhanden)
  const buttons = document.querySelectorAll('.navigation button');
  const slides = document.querySelectorAll('.slide');
  const progressBar = document.querySelector('.progress-bar .progress');

  buttons.forEach((button, index) => {
    button.addEventListener('click', function () {
      const target = this.getAttribute('data-target');

      // Aktivieren der entsprechenden Folie
      slides.forEach(slide => {
        slide.classList.toggle('active', slide.id === target);
      });

      // Fortschrittsbalken aktualisieren
      progressBar.style.width = `${((index + 1) / buttons.length) * 100}%`;
    });
  });
});