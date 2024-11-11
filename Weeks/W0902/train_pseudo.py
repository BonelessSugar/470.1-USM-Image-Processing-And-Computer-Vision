#CNN pseudo-code training phase
prepare training_set, validation_set
design own network
select optimizer and loss_function

for epoch in range(max_epochs):
    for train_batch_data, train_batch_labels in training_set:
        predictions = forward_pass(network, train_batch_data)
        loss = loss_function(predictions, train_batch_labels)

        gradients = backward_pass(loss, network)
        update_weights(network, gradients, optimizer)

    #validation, optional
    if epoch % 10 == 0:
        for val_batch_data, val_batch_labels in data_loader(val_set...)
        ...
