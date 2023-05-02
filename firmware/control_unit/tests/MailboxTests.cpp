//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../privileged/Mailbox.hpp"

#include <model/ModelHal.hpp>

#include <catch.hpp>

using namespace ethosn;
using namespace control_unit;

/// Tests the behaviour of the ethosn_queue struct and associated functions in the ethosn_firmware.h header.
/// Although that file is in the kernel-module, it is used equally by the kernel and the firmware and
/// we don't have a good unit test framework in the kernel, so it is easier to test it here.
TEST_CASE("Mailbox Queue")
{
    uint32_t writePending = 0;
    // Create a queue
    constexpr uint32_t capacity = 8;
    uint8_t storage[sizeof(ethosn_queue) + capacity];
    ethosn_queue* queue = reinterpret_cast<ethosn_queue*>(storage);
    memset(queue, 0, sizeof(*queue));
    queue->capacity = capacity;

    // Check the queue is reported as empty.
    REQUIRE(ethosn_queue_get_size(queue) == 0);
    REQUIRE(ethosn_queue_get_free_space(queue) == 7);

    // Reading should be an error, and leave the read pointer untouched
    {
        uint8_t readBuffer[2];
        REQUIRE(ethosn_queue_read(queue, readBuffer, sizeof(readBuffer)) == false);
        REQUIRE(queue->read == 0);
    }

    // Write some data, using multiple buffers
    {
        uint8_t writeBuffer0[3]        = { 1, 2, 3 };
        uint8_t writeBuffer1[2]        = { 4, 5 };
        const uint8_t* writeBuffers[2] = { &writeBuffer0[0], &writeBuffer1[0] };
        const uint32_t writeSizes[2]   = { sizeof(writeBuffer0), sizeof(writeBuffer1) };
        REQUIRE(ethosn_queue_write(queue, &writeBuffers[0], writeSizes, 2, &writePending) == true);
        // Update write pointer
        queue->write = writePending;
        REQUIRE((queue->data[0] == 1 && queue->data[1] == 2 && queue->data[2] == 3 && queue->data[3] == 4 &&
                 queue->data[4] == 5));
        REQUIRE(ethosn_queue_get_size(queue) == 5);
        REQUIRE(ethosn_queue_get_free_space(queue) == 2);
    }

    // Try writing some more that doesn't fit
    {
        uint8_t writeBuffer[3]   = { 1, 2, 3 };
        uint8_t* writeBuffers[1] = { &writeBuffer[0] };
        uint32_t writeSizes[1]   = { sizeof(writeBuffer) };
        REQUIRE(ethosn_queue_write(queue, &writeBuffers[0], writeSizes, 1, &writePending) == false);
        // Update write pointer
        queue->write = writePending;
    }

    // Read some data
    {
        uint8_t readBuffer[3] = {};
        REQUIRE(ethosn_queue_read(queue, readBuffer, sizeof(readBuffer)) == true);
        REQUIRE((readBuffer[0] == 1 && readBuffer[1] == 2 && readBuffer[2] == 3));
        REQUIRE(ethosn_queue_get_size(queue) == 2);
        REQUIRE(ethosn_queue_get_free_space(queue) == 5);
    }

    // Now we should have space to write.
    // This write should "wrap around" to the start of the buffer
    {
        uint8_t writeBuffer[5]   = { 6, 7, 8, 9, 10 };
        uint8_t* writeBuffers[1] = { &writeBuffer[0] };
        uint32_t writeSizes[1]   = { sizeof(writeBuffer) };
        REQUIRE(ethosn_queue_write(queue, &writeBuffers[0], writeSizes, 1, &writePending) == true);
        // Update write pointer
        queue->write = writePending;
        REQUIRE((queue->data[0] == 9 && queue->data[1] == 10 && queue->data[5] == 6 && queue->data[6] == 7 &&
                 queue->data[7] == 8));
        REQUIRE(ethosn_queue_get_size(queue) == 7);
        REQUIRE(ethosn_queue_get_free_space(queue) == 0);
    }

    // Read the remaining data, "wrapping round" to the start of the buffer
    {
        uint8_t readBuffer[7];
        REQUIRE(ethosn_queue_read(queue, readBuffer, sizeof(readBuffer)) == true);
        REQUIRE((readBuffer[0] == 4 && readBuffer[1] == 5 && readBuffer[2] == 6 && readBuffer[3] == 7 &&
                 readBuffer[4] == 8 && readBuffer[5] == 9 && readBuffer[6] == 10));
        REQUIRE(ethosn_queue_get_size(queue) == 0);
        REQUIRE(ethosn_queue_get_free_space(queue) == 7);
    }
}

// Test write two messages in a row.
TEST_CASE("Mailbox write two messages in a row: Pong and Inference Response")
{
    ModelHal model;
    constexpr uint32_t capacity  = 128;
    constexpr uint32_t testValue = 0xABCD;

    ethosn_mailbox mailboxStorage;
    // Don't use request queue in this test, capacity is 0
    uint8_t requestStorage[sizeof(ethosn_queue)]             = {};
    uint8_t responseStorage[sizeof(ethosn_queue) + capacity] = {};

    ethosn_queue* request  = reinterpret_cast<ethosn_queue*>(requestStorage);
    ethosn_queue* response = reinterpret_cast<ethosn_queue*>(responseStorage);
    /* Setup queue sizes */
    response->capacity = capacity;

    mailboxStorage.request  = reinterpret_cast<ethosn_address_t>(requestStorage);
    mailboxStorage.response = reinterpret_cast<ethosn_address_t>(responseStorage);
    mailboxStorage.severity = ETHOSN_LOG_VERBOSE;
    ethosn_message_header* header;

    Mailbox<ModelHal> mailbox(model, &mailboxStorage);

    REQUIRE(mailbox.SendPong() == Mailbox<ModelHal>::Status::OK);
    REQUIRE(response->write == sizeof(ethosn_message_header));
    // This is a message with payload
    REQUIRE(mailbox.SendInferenceResponse(ETHOSN_INFERENCE_STATUS_OK, testValue, 0) == Mailbox<ModelHal>::Status::OK);
    REQUIRE(response->write == 2 * sizeof(ethosn_message_header) + sizeof(ethosn_message_inference_response));

    // Nothing has happened on the request queue
    REQUIRE(request->read == 0);
    REQUIRE(request->write == 0);
    // Nothing has been read yet
    REQUIRE(response->read == 0);

    // Check Pong
    header = reinterpret_cast<ethosn_message_header*>(&responseStorage[sizeof(ethosn_queue) + response->read]);
    REQUIRE(header->type == ETHOSN_MESSAGE_PONG);
    REQUIRE(header->length == 0);
    // Move read pointer
    response->read = static_cast<uint32_t>(sizeof(ethosn_message_header) + header->length);

    // Check Inference Response
    header = reinterpret_cast<ethosn_message_header*>(&responseStorage[sizeof(ethosn_queue) + response->read]);
    REQUIRE(header->type == ETHOSN_MESSAGE_INFERENCE_RESPONSE);
    REQUIRE(header->length == sizeof(ethosn_message_inference_response));
    // Move read pointer
    response->read += static_cast<uint32_t>(sizeof(ethosn_message_header));
    ethosn_message_inference_response* inference =
        reinterpret_cast<ethosn_message_inference_response*>(&responseStorage[sizeof(ethosn_queue) + response->read]);
    REQUIRE(inference->status == ETHOSN_INFERENCE_STATUS_OK);
    REQUIRE(inference->user_argument == testValue);
}

// Test read two messages in a row
TEST_CASE("Mailbox read two messages in a row: Ping and Configure Profiling Ack")
{
    ModelHal model;
    constexpr uint32_t capacity = 128;

    ethosn_mailbox mailboxStorage;
    uint8_t requestStorage[sizeof(ethosn_queue) + capacity] = {};
    // Don't use response queue in this test, capacity is 0
    uint8_t responseStorage[sizeof(ethosn_queue)] = {};

    ethosn_queue* request  = reinterpret_cast<ethosn_queue*>(requestStorage);
    ethosn_queue* response = reinterpret_cast<ethosn_queue*>(responseStorage);
    /* Setup queue sizes */
    request->capacity = capacity;

    mailboxStorage.request  = reinterpret_cast<ethosn_address_t>(requestStorage);
    mailboxStorage.response = reinterpret_cast<ethosn_address_t>(responseStorage);
    mailboxStorage.severity = ETHOSN_LOG_VERBOSE;
    ethosn_message_header* header;

    Mailbox<ModelHal> mailbox(model, &mailboxStorage);

    // Nothing has been written yet
    REQUIRE(request->write == 0);

    // Send Configure Profiling Ack message
    header         = reinterpret_cast<ethosn_message_header*>(&requestStorage[sizeof(ethosn_queue) + request->write]);
    header->type   = ETHOSN_MESSAGE_CONFIGURE_PROFILING_ACK;
    header->length = 0;
    // Move the write pointer here for simplicity, it's all on the same CPU anyway
    request->write += static_cast<uint32_t>(sizeof(ethosn_message_header));
    request->write += header->length;

    // Send Ping
    header         = reinterpret_cast<ethosn_message_header*>(&requestStorage[sizeof(ethosn_queue) + request->write]);
    header->type   = ETHOSN_MESSAGE_PING;
    header->length = 0;
    request->write += static_cast<uint32_t>(sizeof(ethosn_message_header));

    // Message header storage
    ethosn_message_header headerRx;
    uint8_t buffer[256];

    // Read first message
    REQUIRE(mailbox.ReadMessage(headerRx, buffer, sizeof(buffer)) == Mailbox<ModelHal>::Status::OK);
    // It is a Time Sync
    REQUIRE(headerRx.type == ETHOSN_MESSAGE_CONFIGURE_PROFILING_ACK);
    REQUIRE(headerRx.length == 0);
    // Read second message
    REQUIRE(mailbox.ReadMessage(headerRx, buffer, sizeof(buffer)) == Mailbox<ModelHal>::Status::OK);
    // It's a Ping
    REQUIRE(headerRx.type == ETHOSN_MESSAGE_PING);
    REQUIRE(headerRx.length == 0);
    REQUIRE(request->read == request->write);
    // Nothing has happened on the response queue
    REQUIRE(response->read == 0);
    REQUIRE(response->write == 0);
}
