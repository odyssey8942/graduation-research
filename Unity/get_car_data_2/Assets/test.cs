using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class test : MonoBehaviour
{
    public GameObject xbar, zbar;
    // Use this for initialization
    void Start()
    {
        for (int i = -100; i < 100; i++)
        {
            Instantiate(zbar, new Vector3(1.0f * i * 100, 0.1f, 0.0f), Quaternion.identity);
            Instantiate(xbar, new Vector3(0.0f, 0.1f, 1.0f * i * 100), Quaternion.identity);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}
