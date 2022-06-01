from  torch import optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.NLLLoss()

train_losses=[]
test_losses=[]
acc_list = []

num_epochs=300

for epoch in range(num_epochs):
    
    #print(f"Epoch: {epoch+1}/{num_epochs}")
    train_batch_loss=[]
    for images, labels in iter(train_loader):
        #print(images.shape, labels.shape)
        images=images.to(device)
        labels=labels.to(device)
        
        
        optimizer.zero_grad()
        
        output = model.forward(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        train_batch_loss.append(loss.item())

    mean_train_batch_loss=sum(train_batch_loss)/len(train_batch_loss)
    train_losses.append(mean_train_batch_loss)
    
    model.eval()
    with torch.no_grad():
      batch_test_loss=[]
      for images, labels in iter(test_loader):
        images, labels= images.to(device), labels.to(device)
        
        probability = F.softmax(model(images), dim=1)
        pred = probability.argmax(dim=1)
        test_loss=(pred, labels)
        acc = (pred == labels).sum() / len(labels) * 100
        acc_list.append(acc)
        batch_test_loss.append(test_loss)
      
      mean_test_b_loss=sum(batch_test_loss)/len(batch_test_loss)
      test_losses.append(mean_test_b_loss)
      model.train()
