// Accident Prediction Application - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Form validation enhancement
    const predictionForm = document.getElementById('predictionForm');
    
    if (predictionForm) {
        // Set default values for numeric inputs
        document.getElementById('time_of_day').value = new Date().getHours();
        document.getElementById('traffic_density').value = 5;
        document.getElementById('visibility_meters').value = 5000;
        document.getElementById('temperature_celsius').value = 20;
        
        // Custom form validation
        predictionForm.addEventListener('submit', function(e) {
            // Form is already validated by the built-in HTML5 validation
            // Add any additional custom validation here
            
            // Show loading indicator
            const submitButton = this.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            submitButton.disabled = true;
            
            // Restore button state after prediction is complete
            fetch('/predict', {
                method: 'POST',
                body: new FormData(this)
            })
            .then(response => response.json())
            .then(data => {
                // The main prediction logic is handled in the inline script
                // in index.html for simplicity
                
                // Reset button state
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
                
                // Add animation to the result
                const resultDiv = document.getElementById('predictionResult');
                if (resultDiv) {
                    resultDiv.style.animation = 'fadeIn 0.5s';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
                alert('An error occurred during prediction. Please try again.');
            });
        });
    }
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Enhance form with weather-based suggestions
    const weatherSelect = document.getElementById('weather_condition');
    const roadSelect = document.getElementById('road_condition');
    const visibilityInput = document.getElementById('visibility_meters');
    
    if (weatherSelect && roadSelect && visibilityInput) {
        weatherSelect.addEventListener('change', function() {
            // Suggest road condition based on weather
            const weather = this.value;
            
            switch(weather) {
                case 'rain':
                    roadSelect.value = 'wet';
                    visibilityInput.value = 2000;
                    break;
                case 'snow':
                    roadSelect.value = 'snowy';
                    visibilityInput.value = 1000;
                    break;
                case 'fog':
                    visibilityInput.value = 500;
                    break;
                case 'clear':
                    roadSelect.value = 'dry';
                    visibilityInput.value = 5000;
                    break;
            }
        });
    }
    
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Alt + P for prediction (submit form)
        if (e.altKey && e.key === 'p' && predictionForm) {
            e.preventDefault();
            predictionForm.querySelector('button[type="submit"]').click();
        }
        
        // Alt + D to navigate to dashboard
        if (e.altKey && e.key === 'd') {
            e.preventDefault();
            window.location.href = '/dashboard';
        }
        
        // Alt + H to navigate to home
        if (e.altKey && e.key === 'h' && window.location.pathname !== '/') {
            e.preventDefault();
            window.location.href = '/';
        }
    });
}); 