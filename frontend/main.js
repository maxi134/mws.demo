document.getElementById('submit').addEventListener('click', async () => {
    const inputText = document.getElementById('inputText').value.trim();
    if (!inputText) {
        alert("请输入文本！");
        return;
    }

    console.log('Sending request...');

    try {
        const response = await fetch('http://localhost:5000/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', // 发送 JSON 格式的数据
            },
            body: JSON.stringify({ sentence: inputText }), // 包装数据为 JSON 格式
        });

        console.log('Response status:', response.status); // 打印响应状态
        console.log('Response headers:', response.headers); // 打印响应头

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Response data:', result); // 打印响应数据
        document.getElementById('output').textContent = JSON.stringify(result.prediction, null, 2); // 显示预测结果
    } catch (error) {
        console.error('Error:', error);
        alert('请求失败，请检查您的输入或网络连接！');
    }
});
