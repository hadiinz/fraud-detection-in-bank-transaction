document.getElementById('transaction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const step = document.getElementById('step').value;
    const customer = document.getElementById('customer').value;
    const age = document.getElementById('age').value;
    const gender = document.getElementById('gender').value;
    const zipcodeOri = document.getElementById('zipcodeOri').value;
    const merchant = document.getElementById('merchant').value;
    const zipMerchant = document.getElementById('zipMerchant').value;
    const category = document.getElementById('category').value;
    const amount = document.getElementById('amount').value;

    const transactionData = {
        data: [{
            step: parseInt(step),
            customer: customer,
            age: parseInt(age),
            gender: gender,
            zipcodeOri: zipcodeOri,
            merchant: merchant,
            zipMerchant: zipMerchant,
            category: category,
            amount: parseFloat(amount),
            fraud: 0 // Set default or leave out if not needed
        }]
    };

    // Function to fetch predictions
    const fetchPrediction = (model) => {
        return fetch(`http://127.0.0.1:5000/predict/${model}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        });
    };

    // Fetch predictions for KNN
    fetchPrediction('knn')
    .then(data => {
        document.getElementById('knn-prediction').innerText = data[0].prediction === 1 ? 'Fraud' : 'Not Fraud';
    })
    .catch(error => console.error('Error fetching KNN prediction:', error));

    // Fetch predictions for Random Forest
    fetchPrediction('randomforest')
    .then(data => {
        document.getElementById('rf-prediction').innerText = data[0].prediction === 1 ? 'Fraud' : 'Not Fraud';
    })
    .catch(error => console.error('Error fetching Random Forest prediction:', error));

    // Fetch predictions for XGBoost
    fetchPrediction('xgboost')
    .then(data => {
        document.getElementById('xgboost-prediction').innerText = data[0].prediction === 1 ? 'Fraud' : 'Not Fraud';
    })
    .catch(error => console.error('Error fetching XGBoost prediction:', error));

    // Fetch predictions for Ensemble
    fetchPrediction('ensemble')
    .then(data => {
        document.getElementById('ensemble-prediction').innerText = data[0].prediction === 1 ? 'Fraud' : 'Not Fraud';
    })
    .catch(error => console.error('Error fetching Ensemble prediction:', error));
});

// Sample data
const sampleRequests = {
    fraud: {
        step: 88,
        customer: "C583110837",
        age: 3,
        gender: "M",
        zipcodeOri: "28007",
        merchant: "M480139044",
        zipMerchant: "28007",
        category: "es_health",
        amount: 4449.26,
        fraud: 1
    },
    nonFraud: {
        step: 0,
        customer: "C1093826151",
        age: 4,
        gender: "M",
        zipcodeOri: "28007",
        merchant: "M348934600",
        zipMerchant: "28007",
        category: "es_transportation",
        amount: 4.55,
        fraud: 0
    }
};

// Fill form with sample data
document.querySelectorAll('.sample-btn').forEach(button => {
    button.addEventListener('click', function() {
        const type = this.dataset.type;
        const sampleData = type === 'fraud' ? sampleRequests.fraud : sampleRequests.nonFraud;

        document.getElementById('step').value = sampleData.step;
        document.getElementById('customer').value = sampleData.customer;
        document.getElementById('age').value = sampleData.age;
        document.getElementById('gender').value = sampleData.gender;
        document.getElementById('zipcodeOri').value = sampleData.zipcodeOri;
        document.getElementById('merchant').value = sampleData.merchant;
        document.getElementById('zipMerchant').value = sampleData.zipMerchant;
        document.getElementById('category').value = sampleData.category;
        document.getElementById('amount').value = sampleData.amount;
    });
});
