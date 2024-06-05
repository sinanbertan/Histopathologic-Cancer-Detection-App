document.getElementById('upload-form').addEventListener('submit', function(event) {
    const fileInput = document.getElementById('file-input');
    const alertMessage = document.getElementById('alert-message');
    
    if (!fileInput.files || !fileInput.files[0]) {
        event.preventDefault();  // Prevent the form from submitting if no file is selected
        alert('Lütfen bir resim dosyası seçin.');
    } else {
        // Show success message
        alertMessage.style.display = 'block';
        alertMessage.textContent = 'Dosya başarıyla yüklendi';

        // Allow the form to be submitted
        setTimeout(function() {
            alertMessage.style.display = 'none';
        }, 3000); // Hide the message after 3 seconds
    }
});

// input type="file" element with id="file-input" e bir dosya seçildiğinde, formun submit eventini dinleyen bir event listener ekleyin. Dosya seçilmediyse, formun gönderilmesini engel